"""Run a trained model on any audio or video file, of any length.

    python -m src.infer path/to/file.mp3 [--run multilabel_150]

The model learned from exactly 5 s pictures, so a long file is cut into 5 s
windows, one starting every second (they overlap, so a sound on a window's edge
lands fully inside a neighbour). Each window is predicted separately.

Scores: a multi-label run gives every sound its own 0-1 score. If the run has a
calibration.json (src/calibrate.py), the scores are corrected with it first, so
"60%" means the sound really is there about 60% of the time. Each score then
gets a level:
  - heard:  corrected score >= 0.5 (more likely there than not)
  - maybe:  between the run's chosen cut-off (0.26 for multilabel_150) and 0.5
  - below that the sound is not shown at all
A run without calibration.json uses its plain scores, with no "maybe" band.

Output:
  - overall:  every sound heard or maybe anywhere in the file, with its highest
              score in any window ("did this sound happen anywhere?")
  - timeline: one lane per sound, with the stretches where it was heard or
              maybe ("when did it happen?"); lanes can overlap in time
  - window_times / window_probs: every window's full (corrected) prediction,
              for the web player's live "now hearing" panel
Files shorter than 5 s are padded with silence, as ESC-50 does with short clips.

The live camera mode will reuse `Tagger.predict`, so there is one copy of this.
"""

import argparse
import json
import time

import numpy as np
import torch

from src import audio
from src.models.cnn import SoundCNN
from src.train import RUNS

WINDOW_S = 5.0
HOP_S = 1.0
HEARD = 0.5  # corrected score from which a sound counts as heard


class Tagger:
    """A trained run, loaded once: weights + its own labels, normalisation and correction."""

    def __init__(self, run: str = "multilabel_150", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(RUNS / run / "model.pt", map_location=self.device, weights_only=False)
        self.run = run
        self.labels = ckpt["labels"]
        self.norm = ckpt["norm"]
        self.multilabel = ckpt["config"].get("multilabel", False)
        self.model = SoundCNN(len(self.labels), dropout=ckpt["config"].get("dropout", 0.0),
                              multilabel=self.multilabel)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()  # eval: dropout off, BatchNorm uses saved averages

        # The correction and cut-off chosen on a practice fold, if this run has them.
        path = RUNS / run / "calibration.json"
        self.calibration = json.loads(path.read_text()) if self.multilabel and path.exists() else None
        self.maybe_cut = self.calibration["cut_corrected"] if self.calibration else HEARD

    def level(self, prob: float) -> str | None:
        """'heard', 'maybe', or None (not shown)."""
        return "heard" if prob >= HEARD else "maybe" if prob >= self.maybe_cut else None

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised pictures -> (clips, classes) scores, corrected if the run has a correction."""
        z = self.model(x)
        if not self.multilabel:
            return torch.softmax(z, dim=1)
        if self.calibration:
            z = self.calibration["a"] * z + self.calibration["b"]
        return torch.sigmoid(z)

    def window_starts(self, n_samples: int) -> list[int]:
        """Start sample of every 5 s window, 1 s apart, with the file's end covered."""
        win, hop = int(WINDOW_S * audio.SAMPLE_RATE), int(HOP_S * audio.SAMPLE_RATE)
        if n_samples <= win:
            return [0]
        starts = list(range(0, n_samples - win + 1, hop))
        if starts[-1] + win < n_samples:  # a tail shorter than one hop
            starts.append(n_samples - win)
        return starts

    @torch.no_grad()
    def predict(self, wave: np.ndarray, batch_size: int = 64) -> dict:
        t0 = time.time()
        sr = audio.SAMPLE_RATE
        win = int(WINDOW_S * sr)
        duration = len(wave) / sr
        if len(wave) < win:
            wave = np.pad(wave, (0, win - len(wave)))  # silence, like ESC-50's short clips
        starts = self.window_starts(len(wave))

        # Each window becomes its own picture, exactly as each training clip did.
        pictures = np.stack([audio.log_mel(wave[s:s + win]) for s in starts])
        x = torch.from_numpy((pictures - self.norm["mean"]) / self.norm["std"]).unsqueeze(1)
        probs = torch.cat([
            self.scores(x[i:i + batch_size].float().to(self.device)).cpu()
            for i in range(0, len(x), batch_size)
        ]).numpy()  # (windows, classes)

        best = probs.max(axis=0)
        overall = [{"label": self.labels[i], "prob": round(float(best[i]), 3),
                    "level": self.level(best[i])}
                   for i in np.argsort(best)[::-1] if self.level(best[i])]
        return {
            "duration": round(duration, 2),
            "windows": len(starts),
            "heard_cut": HEARD,
            "maybe_cut": round(self.maybe_cut, 3),
            "calibrated": self.calibration is not None,
            "overall": overall,
            "timeline": self._lanes(probs, starts, duration),
            # Every window's full prediction, placed at its centre, so a player can
            # show "what is sounding now" while the file plays.
            "labels": self.labels,
            "window_times": [round(s / sr + WINDOW_S / 2, 2) for s in starts],
            "window_probs": np.round(probs, 3).tolist(),
            "model": self.run,
            "device": self.device.type,
            "seconds": round(time.time() - t0, 2),
        }

    def _lanes(self, probs, starts, duration) -> list[dict]:
        """One lane per sound that was heard or maybe somewhere, most confident first.

        Windows overlap, so each one is placed at its centre and given the second
        around it (the first from 0, the last up to the file's end). In each lane,
        neighbouring windows with the same level are merged into one segment.
        """
        sr = audio.SAMPLE_RATE
        spans = []
        for k, s in enumerate(starts):
            centre = s / sr + WINDOW_S / 2
            spans.append((0.0 if k == 0 else min(centre - HOP_S / 2, duration),
                          duration if k == len(starts) - 1 else min(centre + HOP_S / 2, duration)))

        lanes = []
        for c, label in enumerate(self.labels):
            segments = []
            for (start, end), p in zip(spans, probs[:, c]):
                level = self.level(p)
                if level is None:
                    continue
                last = segments[-1] if segments else None
                if last and last["level"] == level and abs(last["end"] - start) < 1e-6:
                    last["end"] = end
                    last["probs"].append(float(p))
                else:
                    segments.append({"start": start, "end": end, "level": level, "probs": [float(p)]})
            if segments:
                lanes.append({"label": label, "peak": round(float(probs[:, c].max()), 3),
                              "segments": [{"start": round(g["start"], 2), "end": round(g["end"], 2),
                                            "level": g["level"],
                                            "prob": round(float(np.mean(g["probs"])), 3)}
                                           for g in segments]})
        return sorted(lanes, key=lambda lane: -lane["peak"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag the sounds in any audio/video file.")
    parser.add_argument("path")
    parser.add_argument("--run", default="multilabel_150", help="folder under runs/")
    args = parser.parse_args()

    tagger = Tagger(args.run)
    result = tagger.predict(audio.load_any(args.path))
    print(f"{args.path}: {result['duration']:.1f} s, {result['windows']} windows, "
          f"model {result['model']} on {result['device']}, {result['seconds']:.2f} s")
    print(f"heard: score >= {result['heard_cut']:.2f} | maybe: >= {result['maybe_cut']:.2f}"
          + ("" if result["calibrated"] else " (no calibration.json: plain scores)") + "\n")
    print("overall (highest score in any window):")
    if not result["overall"]:
        print("  nothing it recognises")
    for row in result["overall"]:
        print(f"  {row['label']:18s} {row['prob']:.3f} {row['level']:5s} {'#' * round(row['prob'] * 30)}")
    print("\ntimeline (one lane per sound, mean score per stretch):")
    for lane in result["timeline"]:
        stretches = ", ".join(f"{g['start']:.0f}-{g['end']:.0f}s"
                              + (" (maybe)" if g["level"] == "maybe" else "")
                              for g in lane["segments"])
        print(f"  {lane['label']:18s} {stretches}")


if __name__ == "__main__":
    main()

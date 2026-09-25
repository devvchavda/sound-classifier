"""Run a trained model on any audio or video file, of any length.

    python -m src.infer path/to/file.mp3 [--run with_dropout]

The model learned from exactly 5 s pictures, so a long file is cut into 5 s
windows, one starting every second (they overlap, so a sound on a window's edge
lands fully inside a neighbour). Each window is predicted separately:
  - overall:  for each class, its highest probability in any window
              ("did this sound happen anywhere in the file?")
  - timeline: the winning class of each window, with neighbouring windows that
              agree merged into one segment ("when did it happen?")
  - window_times / window_probs: every window's full prediction, for the web
              player's live "now hearing" panel
Files shorter than 5 s are padded with silence, as ESC-50 does with short clips.

The live camera mode will reuse `Tagger.predict`, so there is one copy of this.
"""

import argparse
import time

import numpy as np
import torch

from src import audio
from src.models.cnn import SoundCNN
from src.train import RUNS

WINDOW_S = 5.0
HOP_S = 1.0
TOP_K = 10


class Tagger:
    """A trained run, loaded once: weights + its own labels + its own normalisation."""

    def __init__(self, run: str = "with_dropout", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(RUNS / run / "model.pt", map_location=self.device, weights_only=False)
        self.run = run
        self.labels = ckpt["labels"]
        self.norm = ckpt["norm"]
        self.model = SoundCNN(len(self.labels), dropout=ckpt["config"].get("dropout", 0.0))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()  # eval: dropout off, BatchNorm uses saved averages

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
            self.model.predict_proba(x[i:i + batch_size].float().to(self.device)).cpu()
            for i in range(0, len(x), batch_size)
        ]).numpy()  # (windows, classes); each row sums to 1

        best = probs.max(axis=0)
        overall = [{"label": self.labels[i], "prob": float(best[i])}
                   for i in np.argsort(best)[::-1][:TOP_K]]
        return {
            "duration": round(duration, 2),
            "windows": len(starts),
            "overall": overall,
            "timeline": self._segments(probs, starts, duration),
            # Every window's full prediction, placed at its centre, so a player can
            # show "what is sounding now" while the file plays.
            "labels": self.labels,
            "window_times": [round(s / sr + WINDOW_S / 2, 2) for s in starts],
            "window_probs": np.round(probs, 3).tolist(),
            "model": self.run,
            "device": self.device.type,
            "seconds": round(time.time() - t0, 2),
        }

    def _segments(self, probs, starts, duration) -> list[dict]:
        """Winning class per window -> merged segments on the file's own clock.

        Windows overlap, so each one is placed at its centre and given the second
        around it; the first segment starts at 0 and the last ends at the file end.
        """
        sr = audio.SAMPLE_RATE
        segments = []
        for k, (s, p) in enumerate(zip(starts, probs)):
            centre = s / sr + WINDOW_S / 2
            start = 0.0 if k == 0 else min(centre - HOP_S / 2, duration)
            end = duration if k == len(starts) - 1 else min(centre + HOP_S / 2, duration)
            label, prob = self.labels[int(p.argmax())], float(p.max())
            if segments and segments[-1]["label"] == label:
                segments[-1]["end"] = end
                segments[-1]["probs"].append(prob)
            else:
                segments.append({"start": start, "end": end, "label": label, "probs": [prob]})
        return [{"start": round(g["start"], 2), "end": round(g["end"], 2), "label": g["label"],
                 "prob": round(float(np.mean(g["probs"])), 3)} for g in segments]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag the sounds in any audio/video file.")
    parser.add_argument("path")
    parser.add_argument("--run", default="with_dropout", help="folder under runs/")
    args = parser.parse_args()

    tagger = Tagger(args.run)
    result = tagger.predict(audio.load_any(args.path))
    print(f"{args.path}: {result['duration']:.1f} s, {result['windows']} windows, "
          f"model {result['model']} on {result['device']}, {result['seconds']:.2f} s\n")
    print("overall (highest probability in any window):")
    for row in result["overall"]:
        print(f"  {row['label']:18s} {row['prob']:.3f} {'#' * round(row['prob'] * 30)}")
    print("\ntimeline (winning class, mean probability):")
    for seg in result["timeline"]:
        print(f"  {seg['start']:6.1f}-{seg['end']:6.1f} s  {seg['label']:18s} {seg['prob']:.2f}")


if __name__ == "__main__":
    main()

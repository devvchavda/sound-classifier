"""Which training clip does a new sound look like, to the model?

Just before its last layer, the CNN sums every 5 s picture up as 256 numbers:
how strongly each of its 256 detectors fired (`Tagger.embed`). That list is
called an *embedding*. Two clips whose embeddings point the same way are clips
the model finds alike, even when a dog barks at second 1 in one and at second
4 in the other, which a pixel-by-pixel comparison of the pictures would miss.
"Point the same way" is measured with cosine similarity: 1 = same direction.

Only the run's TRAINING clips are searched, never its test fold (or practice
fold): these are the examples the model actually learned the sound from.

The same embeddings pick a sound's "typical" examples for its page on the
site: the training clips closest to the average of all its training clips.
"""

import base64
from functools import lru_cache

import numpy as np

from src import audio
from src.datasets import all_clips, custom_meta, esc50_meta, fit_5s
from src.infer import WINDOW_S, Tagger
from src.pictures import log_mel_png


class TrainingClips:
    """The run's training clips: pictures, embeddings, and where each came from."""

    def __init__(self, tagger: Tagger):
        X, y, fold, labels, _ = all_clips()
        if labels != tagger.labels:
            raise ValueError(f"{tagger.run} was trained on other sounds than data/ now holds.")
        # Same order as all_clips: ESC-50 first, then data/custom.
        paths = np.array(list(esc50_meta()["path"]) + list(custom_meta()["path"]))
        test = fold == tagger.config["fold"]
        practice = fold == tagger.config.get("val_fold")
        train = ~test & ~practice
        sources = np.array([source_of(p) for p in paths])

        # Per sound: clips trained on / held out, and how many original recordings.
        self.counts = {label: {"train": int((train & (y == c)).sum()),
                               "test": int((test & (y == c)).sum()),
                               "practice": int((practice & (y == c)).sum()),
                               "recordings": len(set(sources[y == c]))}
                       for c, label in enumerate(labels)}
        self.tagger = tagger
        self.pictures, self.y, self.paths = X[train], y[train], paths[train]
        self.sources = sources[train]
        emb = tagger.embed(self.pictures)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # length 1: dot = cosine

    def nearest(self, label: str, emb: np.ndarray) -> int:
        """Index of the training clip of `label` whose embedding is closest to `emb`."""
        mine = np.flatnonzero(self.y == self.tagger.labels.index(label))
        sims = self.emb[mine] @ (emb / np.linalg.norm(emb))
        return int(mine[np.argmax(sims)])

    def count(self, label: str) -> int:
        return int((self.y == self.tagger.labels.index(label)).sum())

    def typical(self, label: str, n: int = 3) -> list[int]:
        """`n` training clips of `label` closest to its average embedding, each
        from a different original recording (ESC-50 cuts several clips from one)."""
        mine = np.flatnonzero(self.y == self.tagger.labels.index(label))
        centre = self.emb[mine].mean(axis=0)
        picked, seen = [], set()
        for i in mine[np.argsort(-(self.emb[mine] @ centre))]:
            if self.sources[i] not in seen:
                picked.append(int(i))
                seen.add(self.sources[i])
            if len(picked) == n:
                break
        return picked

    def name(self, i: int) -> str:
        return self.paths[i].replace("\\", "/").split("/")[-1]

    @lru_cache(maxsize=256)
    def png(self, i: int) -> bytes:
        return log_mel_png(self.pictures[i])

    def wav(self, i: int) -> np.ndarray:
        """The 5 s the picture was made from, at 16 kHz (what the model heard)."""
        return fit_5s(audio.load(self.paths[i]))

    def compare(self, wave: np.ndarray, result: dict) -> list[dict]:
        """For every sound heard or maybe in `result` (from `Tagger.predict` on
        `wave`): the 5 s slice where it scored highest, and the training clip of
        that sound most like that slice."""
        if not result["overall"]:
            return []
        sr, win = audio.SAMPLE_RATE, int(WINDOW_S * audio.SAMPLE_RATE)
        wave, starts = self.tagger.windows(wave)
        probs = np.array(result["window_probs"])

        rows = []
        for row in result["overall"]:
            k = int(np.argmax(probs[:, self.tagger.labels.index(row["label"])]))
            rows.append({"label": row["label"], "prob": row["prob"], "level": row["level"],
                         "start": round(starts[k] / sr, 2), "end": round(starts[k] / sr + WINDOW_S, 2),
                         "picture": audio.log_mel(wave[starts[k]:starts[k] + win])})
        embs = self.tagger.embed(np.stack([r["picture"] for r in rows]))

        for r, emb in zip(rows, embs):
            i = self.nearest(r["label"], emb)
            png = base64.b64encode(log_mel_png(r.pop("picture"))).decode()
            r["picture"] = f"data:image/png;base64,{png}"
            r["match"] = {"id": i, "name": self.name(i), "of": self.count(r["label"])}
        return rows


def source_of(path: str) -> str:
    """The original recording a clip was cut from. ESC-50: `<fold>-<source>-<take>-<target>.wav`;
    data/custom: `<source>__<rest>.wav` (for speech, the LibriSpeech speaker)."""
    name = path.replace("\\", "/").split("/")[-1]
    return name.split("__")[0] if "__" in name else name.split("-")[1]

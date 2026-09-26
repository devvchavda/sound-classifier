"""Where the clips are, which fold each one belongs to, and their log-mel pictures."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import audio

ROOT = Path(__file__).resolve().parents[1]
ESC50_DIR = ROOT / "data" / "ESC-50-master"
LOGMEL_CACHE = ROOT / "data" / "cache" / "esc50_logmel.npz"
CUSTOM_DIR = ROOT / "data" / "custom"
CUSTOM_CACHE = ROOT / "data" / "cache" / "custom_logmel.npz"
CLIP_SAMPLES = 5 * audio.SAMPLE_RATE  # every picture covers 5 s, like an ESC-50 clip


def esc50_meta() -> pd.DataFrame:
    """One row per clip: path, fold (1-5), target (0-49), category (e.g. 'dog').

    Always split by the `fold` column, never by shuffling. ESC-50 cuts several
    clips from one original recording and keeps them in the same fold; a random
    split would put near-identical clips in train and test, and the score would
    measure memory, not hearing.
    """
    meta = pd.read_csv(ESC50_DIR / "meta" / "esc50.csv")
    meta["path"] = [str(ESC50_DIR / "audio" / f) for f in meta["filename"]]
    return meta[["path", "fold", "target", "category"]]


def esc50_logmels():
    """Every ESC-50 clip as a log-mel picture -> (X, y, fold, meta).

    X has shape (2000, 64, 501): 2000 clips x 64 mel bands x 501 frames (5 s).
    Training shows each clip to the CNN dozens of times, so the pictures are
    computed once (~30 s) and saved. They are built by `src/audio.py`, so
    after changing that file, delete the cache or training will use stale
    pictures.
    """
    meta = esc50_meta()
    if LOGMEL_CACHE.exists():
        z = np.load(LOGMEL_CACHE)
        return z["X"], z["y"], z["fold"], meta
    X = np.stack([audio.log_mel(audio.load(p)) for p in tqdm(meta["path"], desc="log-mel")])
    LOGMEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LOGMEL_CACHE, X=X, y=meta["target"].values, fold=meta["fold"].values)
    return X, meta["target"].values, meta["fold"].values, meta


def custom_meta() -> pd.DataFrame:
    """One row per file in data/custom/<label>/: path, label, source, fold.

    Any folder there is a class; the folder name is the label. Files are named
    `<source>__<rest>.wav`, and every clip of one source gets the same fold
    (for speech, the source is the LibriSpeech speaker). Same reason as ESC-50's
    folds: if one voice were in both training and test, the model could pass
    the test by recognising that person instead of speech. Within each label
    the sources are sorted and dealt out to folds 1-5 in turn, so the split is
    the same in every run.
    """
    rows = []
    label_dirs = sorted(p for p in CUSTOM_DIR.iterdir() if p.is_dir()) if CUSTOM_DIR.exists() else []
    for label_dir in label_dirs:
        files = sorted(label_dir.glob("*.wav"))
        sources = sorted({f.name.split("__")[0] for f in files})
        fold_of = {s: i % 5 + 1 for i, s in enumerate(sources)}
        for f in files:
            source = f.name.split("__")[0]
            rows.append({"path": str(f), "label": label_dir.name, "source": source,
                         "fold": fold_of[source]})
    return pd.DataFrame(rows, columns=["path", "label", "source", "fold"])


def fit_5s(wave: np.ndarray) -> np.ndarray:
    """Pad a short clip with silence (as ESC-50 does), or keep the middle 5 s of a long one."""
    if len(wave) < CLIP_SAMPLES:
        return np.pad(wave, (0, CLIP_SAMPLES - len(wave)))
    start = (len(wave) - CLIP_SAMPLES) // 2
    return wave[start:start + CLIP_SAMPLES]


def custom_logmels(meta: pd.DataFrame) -> np.ndarray:
    """Log-mel picture of every clip in `meta` -> (clips, 64, 501), cached.

    The cache remembers which files it was built from, so dropping new clips
    into data/custom/ rebuilds it instead of silently using the old pictures.
    """
    names = np.array([Path(p).relative_to(CUSTOM_DIR).as_posix() for p in meta["path"]])
    if CUSTOM_CACHE.exists():
        z = np.load(CUSTOM_CACHE)
        if np.array_equal(z["names"], names):
            return z["X"]
    X = np.stack([audio.log_mel(fit_5s(audio.load(p)))
                  for p in tqdm(meta["path"], desc="log-mel (custom)")])
    CUSTOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CUSTOM_CACHE, X=X, names=names)
    return X


def all_clips():
    """ESC-50 + data/custom together -> (X, y, fold, labels, is_esc).

    Every clip still holds ONE sound; mixing happens later, while training.
    `labels` is built from the data, never typed in: ESC-50's 50 categories in
    their usual order, then each data/custom/ folder (alphabetical). A custom
    folder named like an ESC-50 class adds clips to that class. `is_esc` marks
    the ESC-50 clips, whose fold-5 top-1 accuracy is comparable with the
    single-label runs.
    """
    X_esc, y_esc, fold_esc, meta = esc50_logmels()
    labels = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    cmeta = custom_meta()
    if len(cmeta) == 0:
        return X_esc, y_esc, fold_esc, labels, np.ones(len(X_esc), dtype=bool)
    labels += [name for name in sorted(cmeta["label"].unique()) if name not in labels]
    X = np.concatenate([X_esc, custom_logmels(cmeta)])
    y = np.concatenate([y_esc, [labels.index(name) for name in cmeta["label"]]])
    fold = np.concatenate([fold_esc, cmeta["fold"].values])
    is_esc = np.arange(len(X)) < len(X_esc)
    return X, y, fold, labels, is_esc

"""Where the clips are, which fold each one belongs to, and their log-mel pictures."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import audio

ROOT = Path(__file__).resolve().parents[1]
ESC50_DIR = ROOT / "data" / "ESC-50-master"
LOGMEL_CACHE = ROOT / "data" / "cache" / "esc50_logmel.npz"


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

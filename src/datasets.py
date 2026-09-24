"""Where the clips are, and which fold each one belongs to."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESC50_DIR = ROOT / "data" / "ESC-50-master"


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

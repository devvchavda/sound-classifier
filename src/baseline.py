"""Baseline: hand-crafted MFCC statistics + classic scikit-learn models.

    python -m src.baseline

This is the "simple thing". Each 5 s clip is squashed into ONE fixed-length
vector (the average and spread of every MFCC over time), then a Logistic
Regression / Random Forest learns from those vectors.

What it throws away is exactly why we later need a CNN: averaging over time
forgets *order* ("bark ... bark" and "hum" can average out to similar numbers).
Its score is the bar our own model has to clear.
"""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import audio
from src.datasets import ROOT, esc50_meta

import matplotlib.pyplot as plt

CACHE = ROOT / "data" / "cache" / "baseline_features.npz"
FIGURES = ROOT / "reports" / "figures"


def clip_features(wave: np.ndarray) -> np.ndarray:
    """One clip -> 80 numbers (see `mfcc_stats`)."""
    return mfcc_stats(audio.mfcc(wave))


def mfcc_stats(m: np.ndarray) -> np.ndarray:
    """MFCCs (20, frames) -> 80 numbers: mean & std over time of the 20 MFCCs
    and of their deltas (how fast each MFCC is changing)."""
    d = np.diff(m, axis=1)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)])


def load_features():
    meta = esc50_meta()
    if CACHE.exists():
        z = np.load(CACHE)
        return z["X"], z["y"], z["fold"], meta
    X = np.stack([clip_features(audio.load(p)) for p in tqdm(meta["path"], desc="MFCC")])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, y=meta["target"].values, fold=meta["fold"].values)
    return X, meta["target"].values, meta["fold"].values, meta


MODELS = {
    # StandardScaler: puts every feature on the same scale, otherwise features
    # with big numbers dominate the linear model's decisions.
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)
    ),
    "random_forest": lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0),
}


def cross_validate(name: str, X: np.ndarray, y: np.ndarray, fold: np.ndarray):
    """5-fold CV for one model -> (per-fold accuracies, predictions for every clip).

    Each fold takes one turn as the unseen test set, so every clip is tested
    exactly once and the score doesn't depend on one lucky split.
    `y_pred[i]` is the guess for clip i, made by the model that never saw it.
    """
    accs, y_pred = [], np.empty_like(y)
    for k in range(1, 6):
        train, test = fold != k, fold == k
        model = MODELS[name]().fit(X[train], y[train])
        y_pred[test] = model.predict(X[test])
        accs.append(accuracy_score(y[test], y_pred[test]))
    return accs, y_pred


def main() -> None:
    # Save figures to files without opening windows. Set here, not at import,
    # so notebooks that import this module keep their inline plots.
    matplotlib.use("Agg")
    X, y, fold, meta = load_features()
    names = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    FIGURES.mkdir(parents=True, exist_ok=True)

    for name in MODELS:
        accs, y_pred = cross_validate(name, X, y, fold)
        y_true = y
        per_fold = " ".join(f"{a:.3f}" for a in accs)
        print(f"{name:20s} accuracy {np.mean(accs):.3f} ± {np.std(accs):.3f}   folds: {per_fold}")

        fig, ax = plt.subplots(figsize=(16, 16))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=names, ax=ax, colorbar=False,
            xticks_rotation="vertical", include_values=False, normalize="true",
        )
        ax.set_title(f"Baseline {name} (MFCC stats) — 5-fold confusion matrix")
        fig.tight_layout()
        fig.savefig(FIGURES / f"baseline_{name}_confusion.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()

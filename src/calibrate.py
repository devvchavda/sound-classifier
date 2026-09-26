"""Choose the "yes" cut-off, and check whether the scores mean what they say.

    python -m src.calibrate multilabel_val4

Needs a run trained with `--val-fold` (a practice-exam fold it never trained
on). Everything is CHOSEN on that fold; fold 5 is only used at the end, once,
to report how the choices hold up on unseen clips.

1. Honesty check: group the practice-exam answers by score (all the ~30%
   answers, all the ~50% ones, ...) and count how often the sound was really
   there. Honest scores: ~30% of the "30%" answers are right.
2. Correction (calibration): fit new_score = sigmoid(a * raw + b) on the
   practice exam, so the corrected numbers match how often the model is right.
   It keeps the order of the scores, so it changes the number shown, never
   which clip ranks above which (mAP is unchanged).
3. Cut-off: the one with the best F1 on the practice exam. F1 is high only
   when precision AND recall are both high, so it is the best balance between
   "finds the sounds" and "doesn't invent sounds".

Saves calibration.json into the run folder for the website to use.

    python -m src.calibrate multilabel_150 --from multilabel_val4

The second form reuses choices already made on another run's practice exam
(for a model trained on more folds, which has no practice exam left). Nothing
is chosen here: fold 5 only reports how well the borrowed correction fits.
"""

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.datasets import all_clips
from src.mixing import Mixer
from src.models.cnn import SoundCNN
from src.train import RUNS
from src.train_multilabel import TEST_MIX_SEED, mix_scores

VAL_MIX_SEED = 4_000
VAL_MIXES = 2_000  # more than the test's 1,000: each score bin needs enough answers
BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@torch.no_grad()
def raw_scores(model, x_db, norm, batch_size=100) -> np.ndarray:
    """Pictures in dB -> the model's raw scores (before sigmoid), shape (clips, classes)."""
    model.eval()
    return torch.cat([model((x_db[i:i + batch_size] - norm["mean"]) / norm["std"])
                      for i in range(0, len(x_db), batch_size)]).cpu().numpy()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def honesty_table(p: np.ndarray, Y: np.ndarray, title: str) -> None:
    """For each score range: how many answers, their average score, how often the sound was there."""
    print(f"\n{title}")
    print(f"  {'score':>9s} | {'answers':>7s} | {'avg score':>9s} | {'really there':>12s}")
    p, Y = p.ravel(), Y.ravel()
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        inside = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if inside.sum() == 0:
            continue
        print(f"  {lo:.1f}-{hi:.1f} | {inside.sum():7d} | {p[inside].mean():9.2f} | "
              f"{Y[inside].mean():12.2f}")


def prf(p: np.ndarray, Y: np.ndarray, cut: float):
    """Precision, recall and F1 when every score >= cut counts as "yes"."""
    yes = p >= cut
    hit = (yes & (Y == 1)).sum()
    precision = hit / max(yes.sum(), 1)
    recall = hit / max(Y.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(precision), float(recall), float(f1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="folder under runs/, trained with --val-fold")
    parser.add_argument("--from", dest="source", default=None,
                        help="reuse this run's calibration.json instead of choosing anew")
    args = parser.parse_args()

    run_dir = RUNS / args.run
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    cfg, labels, norm = ckpt["config"], ckpt["labels"], ckpt["norm"]
    if not cfg.get("val_fold") and not args.source:
        raise SystemExit("This run has no practice-exam fold; train it with --val-fold.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoundCNN(len(labels), multilabel=True).to(device)
    model.load_state_dict(ckpt["state_dict"])

    X, y, fold, _, _ = all_clips()
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).long().to(device)
    fold = torch.from_numpy(fold).to(device)

    def mixes(which_fold, n, seed):
        pool = fold == which_fold
        x, Y = Mixer(X[pool], y[pool], len(labels)).fixed_set(n, seed)
        return raw_scores(model, x, norm), Y.cpu().numpy()

    if args.source:
        report_borrowed(args, run_dir, cfg, mixes)
        return

    z_val, Y_val = mixes(cfg["val_fold"], VAL_MIXES, VAL_MIX_SEED)
    z_test, Y_test = mixes(cfg["fold"], cfg["test_mixes"], TEST_MIX_SEED)
    p_val, p_test = sigmoid(z_val), sigmoid(z_test)
    print(f"practice exam: fold {cfg['val_fold']}, {len(z_val)} mixes | "
          f"final exam: fold {cfg['fold']}, {len(z_test)} mixes")

    # 1. Are the raw scores honest?
    honesty_table(p_val, Y_val, f"1. RAW scores on the practice exam (fold {cfg['val_fold']})")

    # 2. Correction: one a and b shared by all sounds, fitted on the practice exam only.
    fit = LogisticRegression(C=1e6).fit(z_val.reshape(-1, 1), Y_val.ravel())
    a, b = float(fit.coef_[0, 0]), float(fit.intercept_[0])
    print(f"\n2. correction fitted on the practice exam: new = sigmoid({a:.3f} x raw + {b:+.3f})")
    q_test = sigmoid(a * z_test + b)
    honesty_table(q_test, Y_test, f"   CORRECTED scores, checked on the final exam (fold {cfg['fold']})")

    # 3. Cut-off with the best F1 on the practice exam (raw scale; the corrected
    #    scale gives the same yes/no answers, because the correction keeps the order).
    cuts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [prf(p_val, Y_val, c)[2] for c in cuts]
    cut = float(cuts[int(np.argmax(f1s))])
    cut_corrected = float(sigmoid(a * np.log(cut / (1 - cut)) + b))
    print(f"\n3. best cut-off on the practice exam: raw {cut:.2f} "
          f"(= {cut_corrected:.2f} on the corrected scale), F1 {max(f1s):.3f}")

    # Report, once, on the final exam.
    m_raw = mix_scores(torch.from_numpy(p_test), torch.from_numpy(Y_test))["map"]
    m_cor = mix_scores(torch.from_numpy(q_test), torch.from_numpy(Y_test))["map"]
    print(f"\nFINAL EXAM (fold {cfg['fold']}), reported once:")
    print(f"  mAP raw {m_raw:.3f} | corrected {m_cor:.3f} (same ranking, so the same mAP)")
    for name, c in [("cut-off 0.50 (raw)", 0.5), (f"cut-off {cut:.2f} (chosen)", cut)]:
        P, R, F = prf(p_test, Y_test, c)
        print(f"  {name:22s} precision {P:.3f} | recall {R:.3f} | F1 {F:.3f}")

    out = {"a": a, "b": b, "cut_raw": cut, "cut_corrected": cut_corrected,
           "val_fold": cfg["val_fold"], "val_f1": max(f1s)}
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {run_dir / 'calibration.json'}")


def report_borrowed(args, run_dir, cfg, mixes) -> None:
    """Apply another run's correction and cut-off to this run; report on fold 5 only."""
    chosen = json.loads((RUNS / args.source / "calibration.json").read_text())
    a, b, cut = chosen["a"], chosen["b"], chosen["cut_corrected"]
    z_test, Y_test = mixes(cfg["fold"], cfg["test_mixes"], TEST_MIX_SEED)
    p_test, q_test = sigmoid(z_test), sigmoid(a * z_test + b)
    print(f"correction and cut-off borrowed from {args.source} (chosen on its fold "
          f"{chosen['val_fold']}): new = sigmoid({a:.3f} x raw + {b:+.3f}), cut-off {cut:.2f}")
    honesty_table(p_test, Y_test, f"RAW scores on the final exam (fold {cfg['fold']}), for comparison")
    honesty_table(q_test, Y_test, f"BORROWED correction on the final exam (fold {cfg['fold']})")
    m = mix_scores(torch.from_numpy(q_test), torch.from_numpy(Y_test))["map"]
    print(f"\nFINAL EXAM (fold {cfg['fold']}), report only: mAP {m:.3f}")
    for name, c, p in [("raw >= 0.50", 0.5, p_test), ("corrected >= 0.50", 0.5, q_test),
                       (f"corrected >= {cut:.2f}", cut, q_test)]:
        P, R, F = prf(p, Y_test, c)
        print(f"  {name:20s} precision {P:.3f} | recall {R:.3f} | F1 {F:.3f}")
    out = {**chosen, "borrowed_from": args.source}
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {run_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()

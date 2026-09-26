"""Train the multi-label CNN: an independent yes/no score for every sound.

    python -m src.train_multilabel --fold 5 --epochs 60

Same CNN as `src.train`, two changes:
  - the answer: sigmoid + binary cross-entropy instead of softmax. Each of the
    51 sounds (ESC-50's 50 + every data/custom/ folder, e.g. speech) is its own
    "is it there?" question, graded on its own;
  - the data: every batch is freshly mixed from 1-3 single-sound clips of the
    training folds (src/mixing.py), labelled with every sound in it. An epoch
    is 1,600 new mixes; almost none ever repeat.

Scored every epoch on the test fold only (reasons in reports/journal.md):
  - mAP on 1,000 fixed test mixes: for each sound, are the clips that contain
    it ranked above the ones that don't? Averaged over all sounds. No cut-off.
  - precision / recall at 0.5, on the same mixes: of the sounds it says "yes"
    to, how many are there? Of the sounds that are there, how many did it find?
  - top-1 on the 400 single ESC-50 test clips: the old grading, so this model
    can be set next to the single-label runs (0.790 / 0.787).

Reporting rule, as before: the LAST epoch's numbers, decided in advance.

`--val-fold 4` keeps a second fold out of training as a practice exam: the
cut-off and any score correction are chosen on it afterwards
(`python -m src.calibrate <run>`), so fold 5 is never used to choose anything.
"""

import argparse
import json
import time
from datetime import datetime

import matplotlib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.datasets import all_clips
from src.mixing import Mixer
from src.models.cnn import SoundCNN
from src.train import RUNS, TEST_COLOR, TRAIN_COLOR

import matplotlib.pyplot as plt

TEST_MIX_SEED = 5_000        # fixed test mixes: the same exam every epoch and every run
TRAIN_CHECK_SEED = 6_000     # fixed training mixes, only for the final "memorised?" check


@torch.no_grad()
def probabilities(model, x_db, norm, batch_size=100):
    """Pictures in dB -> (clips, classes) probabilities, model in eval mode."""
    model.eval()  # dropout off, BatchNorm uses its saved averages
    return torch.cat([
        model.predict_proba((x_db[i:i + batch_size] - norm["mean"]) / norm["std"])
        for i in range(0, len(x_db), batch_size)
    ])


def mix_scores(probs, labels, threshold=0.5):
    """mAP, precision and recall on a set of mixes; also AP per class."""
    P, Y = probs.cpu().numpy(), labels.cpu().numpy()
    present = Y.sum(axis=0) > 0  # AP needs at least one clip containing the sound
    ap = np.full(Y.shape[1], np.nan)
    ap[present] = average_precision_score(Y[:, present], P[:, present], average=None)
    said_yes = P >= threshold
    true_yes = (said_yes & (Y == 1)).sum()
    return {"map": float(np.nanmean(ap)),
            "precision": float(true_yes / max(said_yes.sum(), 1)),
            "recall": float(true_yes / max(Y.sum(), 1)),
            "ap": ap}


def top1(probs, y):
    """Share of single-sound clips whose highest score is the right sound."""
    return float((probs.argmax(dim=1) == y).float().mean())


def plot_curves(history, title, path):
    """Loss (train vs test) and the test scores, per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, name, color in [("train_loss", "train (mixes)", TRAIN_COLOR),
                             ("test_loss", "test (mixes)", TEST_COLOR)]:
        axes[0].plot(epochs, [h[key] for h in history], color=color, linewidth=2, label=name)
    axes[0].set_title("Loss (binary cross-entropy)", loc="left")
    for key, name, color in [("test_map", "mAP, test mixes", TEST_COLOR),
                             ("test_precision", "precision @0.5", "#7a7a7a"),
                             ("test_recall", "recall @0.5", "#b0b0b0"),
                             ("single_top1", "top-1, single test clips", "#2a9d5c")]:
        axes[1].plot(epochs, [h[key] for h in history], color=color, linewidth=2, label=name)
    axes[1].axhline(0.790, color="#2a9d5c", linestyle=":", linewidth=1)
    axes[1].annotate("single-label model: 0.790", (epochs[0], 0.790), xytext=(0, 4),
                     textcoords="offset points", color="#2a9d5c", fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test scores", loc="left")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=int, default=5, help="test fold (1-5)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="starting learning rate")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mixes-per-epoch", type=int, default=1600)
    parser.add_argument("--test-mixes", type=int, default=1000)
    parser.add_argument("--val-fold", type=int, default=None,
                        help="also keep this fold out of training, as a practice exam "
                             "for choosing the cut-off (src.calibrate)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run folder name (default: fold + time)")
    args = parser.parse_args()

    matplotlib.use("Agg")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All clips stay in decibels on the GPU: mixing has to happen before normalising.
    X, y, fold, labels, is_esc = all_clips()
    train, test = fold != args.fold, fold == args.fold
    if args.val_fold:
        train &= fold != args.val_fold
    norm = {"mean": float(X[train].mean(dtype=np.float64)),
            "std": float(X[train].std(dtype=np.float64))}
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).long().to(device)
    train_t, test_t = torch.from_numpy(train).to(device), torch.from_numpy(test).to(device)

    # Two separate pools: a test sound can never end up inside a training mix.
    train_mixer = Mixer(X[train_t], y[train_t], len(labels))
    test_mixer = Mixer(X[test_t], y[test_t], len(labels))
    x_test, Y_test = test_mixer.fixed_set(args.test_mixes, TEST_MIX_SEED)
    single = test_t & torch.from_numpy(is_esc).to(device)
    x_single, y_single = X[single].unsqueeze(1), y[single]

    speech = labels.index("speech") if "speech" in labels else None
    kept_out = f"{args.fold}" + (f", {args.val_fold}" if args.val_fold else "")
    print(f"device {device} | {len(labels)} sounds | train pool {int(train.sum())} clips "
          f"(folds not {kept_out}) | test pool {int(test.sum())} clips (fold {args.fold})")
    if args.val_fold:
        print(f"fold {args.val_fold} kept out as the practice exam (not trained on, not tested on)")
    print(f"test: {len(x_test)} fixed mixes + {len(x_single)} single ESC-50 clips | "
          f"{args.mixes_per_epoch} fresh training mixes per epoch")

    model = SoundCNN(n_classes=len(labels), dropout=args.dropout, multilabel=True).to(device)
    # Sigmoid + binary cross-entropy in one step: for each of the 51 sounds,
    # "how wrong was your yes/no?", averaged over sounds and clips.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)

    def test_loss():
        model.eval()
        with torch.no_grad():
            return float(sum(loss_fn(model((x_test[i:i + 100] - norm["mean"]) / norm["std"]),
                                     Y_test[i:i + 100]) * len(x_test[i:i + 100])
                             for i in range(0, len(x_test), 100)) / len(x_test))

    # Sanity check: before learning, every score is ~0, i.e. "50/50" on every
    # sound, and binary cross-entropy for a 50/50 guess is log 2 = 0.693.
    print(f"untrained: test loss {test_loss():.3f} (a 50/50 guess = 0.693)\n")

    history = []
    start = time.time()
    steps = args.mixes_per_epoch // args.batch_size
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        total = 0.0
        for _ in range(steps):
            x_db, Y = train_mixer(args.batch_size, generator)
            loss = loss_fn(model((x_db - norm["mean"]) / norm["std"]), Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        scores = mix_scores(probabilities(model, x_test, norm), Y_test)
        single_acc = top1(probabilities(model, x_single, norm), y_single)
        history.append({"epoch": epoch, "lr": lr, "train_loss": total / steps,
                        "test_loss": test_loss(), "test_map": scores["map"],
                        "test_precision": scores["precision"], "test_recall": scores["recall"],
                        "single_top1": single_acc})
        h = history[-1]
        print(f"epoch {epoch:3d}/{args.epochs} | loss train {h['train_loss']:.4f} "
              f"test {h['test_loss']:.4f} | mAP {h['test_map']:.3f} | "
              f"P {h['test_precision']:.3f} R {h['test_recall']:.3f} | "
              f"single top-1 {single_acc:.3f} | lr {lr:.5f} | {time.time() - t0:.1f}s")

    # Memorised? The same scores on TRAINING data, measured clean (no dropout).
    x_check, Y_check = train_mixer.fixed_set(args.test_mixes, TRAIN_CHECK_SEED)
    train_scores = mix_scores(probabilities(model, x_check, norm), Y_check)
    train_single = train_t & torch.from_numpy(is_esc).to(device)
    train_top1 = top1(probabilities(model, X[train_single].unsqueeze(1), norm), y[train_single])

    name = args.name or f"multilabel_fold{args.fold}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "multilabel": True}
    torch.save({"state_dict": model.state_dict(), "labels": labels, "norm": norm,
                "config": config}, run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    per_class = {labels[i]: round(float(a), 4) for i, a in enumerate(scores["ap"])}
    (run_dir / "per_class_ap.json").write_text(json.dumps(per_class, indent=1))
    plot_curves(history, f"Multi-label CNN, test fold {args.fold} ({name})", run_dir / "curves.png")

    ranked = sorted(per_class.items(), key=lambda kv: kv[1])
    print(f"\nfinished in {(time.time() - start) / 60:.1f} min")
    print("hardest sounds (AP): " + ", ".join(f"{k} {v:.2f}" for k, v in ranked[:5]))
    print("easiest sounds (AP): " + ", ".join(f"{k} {v:.2f}" for k, v in ranked[-5:]))
    if speech is not None:
        print(f"speech AP: {per_class['speech']:.3f}")
    print(f"clean TRAIN scores: mAP {train_scores['map']:.3f} | single top-1 {train_top1:.3f}")
    print(f"REPORTED RESULT (last epoch): mAP {history[-1]['test_map']:.3f} | "
          f"P {history[-1]['test_precision']:.3f} R {history[-1]['test_recall']:.3f} @0.5 | "
          f"single-clip top-1 {history[-1]['single_top1']:.3f} (single-label model: 0.790)")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()

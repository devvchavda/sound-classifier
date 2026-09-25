"""Train our CNN on ESC-50: one fold is the unseen test set, the other four train.

    python -m src.train --fold 5 --epochs 60

Each epoch: shuffle the training clips, walk through them in batches of 32
(predict -> loss -> backpropagate -> Adam step), then measure the test fold
without learning anything. Prints one line per epoch and saves a run folder
under runs/ with the weights, labels, normalisation numbers, per-epoch history
and the loss/accuracy curves.

Reporting rule: the number to report is the LAST epoch's test accuracy, decided
in advance. Picking the best-looking epoch would let the test fold choose the
model, and it would stop being unseen.
"""

import argparse
import json
import math
import time
from datetime import datetime

import matplotlib
import numpy as np
import torch
from torch import nn

from src.datasets import ROOT, esc50_logmels
from src.models.cnn import SoundCNN

import matplotlib.pyplot as plt

RUNS = ROOT / "runs"
TRAIN_COLOR, TEST_COLOR = "#2a78d6", "#eb6834"


def load_split(test_fold: int, device: torch.device):
    """Cached log-mels -> normalised tensors on `device`, split by fold.

    Normalisation (one mean, one std) is measured on the training folds only:
    letting the test fold into that average would leak a little of it into
    training. The same two numbers are saved with the model, so the website
    shifts live audio by exactly the same amount.
    """
    X, y, fold, meta = esc50_logmels()
    labels = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    train, test = fold != test_fold, fold == test_fold
    mean = float(X[train].mean(dtype=np.float64))
    std = float(X[train].std(dtype=np.float64))

    def to_tensors(mask):
        # (clips, 64, 501) -> (clips, 1, 64, 501): the CNN expects a channel axis.
        x = torch.from_numpy((X[mask] - mean) / std).unsqueeze(1).float()
        return x.to(device), torch.from_numpy(y[mask]).long().to(device)

    # The whole dataset (~250 MB) fits in GPU memory, so batches are just slices
    # of one big tensor; no loading from disk while training.
    return to_tensors(train), to_tensors(test), labels, {"mean": mean, "std": std}


def train_one_epoch(model, x, y, loss_fn, optimizer, batch_size, generator):
    """One pass over the training clips, in a fresh random order."""
    model.train()  # BatchNorm uses each batch's own average
    order = torch.randperm(len(x), generator=generator).to(x.device)
    total_loss, correct = 0.0, 0
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        scores = model(x[idx])
        loss = loss_fn(scores, y[idx])  # softmax + cross-entropy in one step

        optimizer.zero_grad()  # forget the previous batch's slopes
        loss.backward()        # backpropagation: a slope for every weight
        optimizer.step()       # Adam nudges every weight downhill

        total_loss += loss.item() * len(idx)
        correct += (scores.argmax(dim=1) == y[idx]).sum().item()
    return total_loss / len(x), correct / len(x)


@torch.no_grad()
def evaluate(model, x, y, loss_fn, batch_size=100):
    """Loss and accuracy on clips the model does not learn from."""
    model.eval()  # BatchNorm switches to the averages it saved during training
    total_loss, correct = 0.0, 0
    for start in range(0, len(x), batch_size):
        scores = model(x[start:start + batch_size])
        total_loss += loss_fn(scores, y[start:start + batch_size]).item() * len(scores)
        correct += (scores.argmax(dim=1) == y[start:start + batch_size]).sum().item()
    return total_loss / len(x), correct / len(x)


def plot_curves(history, title, path):
    """Loss and accuracy per epoch, train vs test, side by side."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, name in [(axes[0], "loss", "Loss (cross-entropy)"),
                             (axes[1], "acc", "Accuracy")]:
        for split, color in [("train", TRAIN_COLOR), ("test", TEST_COLOR)]:
            values = [h[f"{split}_{metric}"] for h in history]
            ax.plot(epochs, values, color=color, linewidth=2, label=split)
            ax.annotate(split, (epochs[-1], values[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color="#333333")
        ax.set_title(name, loc="left")
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[1].set_ylim(0, 1)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run folder name (default: fold + time)")
    args = parser.parse_args()

    matplotlib.use("Agg")  # save figures to files, don't open windows
    torch.manual_seed(args.seed)  # same seed -> same starting weights -> comparable runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (x_train, y_train), (x_test, y_test), labels, norm = load_split(args.fold, device)
    print(f"device {device} | train {len(x_train)} clips (folds != {args.fold}) | "
          f"test {len(x_test)} clips (fold {args.fold}) | {len(labels)} classes")
    print(f"normalisation from training folds: mean {norm['mean']:.2f} dB, "
          f"std {norm['std']:.2f} dB")

    model = SoundCNN(n_classes=len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Big steps early, smaller and smaller ones later, so it settles in the valley.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)

    # Sanity check before any learning: a blind guess over 50 classes scores
    # about -log(1/50) = 3.9. Far from that means something is wired wrong.
    loss0, acc0 = evaluate(model, x_test, y_test, loss_fn)
    print(f"untrained: test loss {loss0:.3f} (blind guess = {math.log(len(labels)):.3f}), "
          f"acc {acc0:.3f}\n")

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, x_train, y_train, loss_fn, optimizer, args.batch_size, generator)
        test_loss, test_acc = evaluate(model, x_test, y_test, loss_fn)
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.3f} acc {train_acc:.3f}"
              f" | test loss {test_loss:.3f} acc {test_acc:.3f} | lr {lr:.5f}"
              f" | {time.time() - t0:.1f}s")

    name = args.name or f"fold{args.fold}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Everything needed to rebuild this exact model later (e.g. in the server).
    torch.save({"state_dict": model.state_dict(), "labels": labels, "norm": norm,
                "config": vars(args)}, run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    plot_curves(history, f"Our CNN, test fold {args.fold} ({name})", run_dir / "curves.png")

    print(f"\nfinished in {(time.time() - start) / 60:.1f} min")
    print(f"REPORTED RESULT (last epoch): test accuracy {history[-1]['test_acc']:.3f}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()

"""Our CNN, built from scratch: 4 conv blocks -> global pooling -> dropout -> 50 scores.

    python -m src.models.cnn      # shape check: one real clip through the model

The spectrogram is treated like a one-channel image (64 mel bands x frames).
Small 3x3 filters slide over it and learn local patterns (lines, clicks,
harmonics); each block shrinks the picture so the next block's filters see a
wider stretch of the sound.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """3x3 filters -> BatchNorm -> ReLU -> MaxPool 2x2 (picture halves each way).

    - BatchNorm re-centres each filter's output around 0 with spread ~1, so the
      next block isn't chasing a moving target while this one is still learning.
    - ReLU keeps positives and zeroes negatives. Without it, stacked blocks would
      collapse into one big multiply-and-add, and depth would buy nothing.
    - MaxPool keeps the strongest value in each 2x2 patch: a wider view for the
      next block, tolerance to small shifts, and less compute.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class SoundCNN(nn.Module):
    """(batch, 1, 64 mels, frames) -> (batch, n_classes) raw scores.

    The output is raw scores, not probabilities: softmax is applied by the loss
    during training and by `predict_proba` when we want probabilities.
    """

    def __init__(self, n_classes: int, dropout: float = 0.5):
        super().__init__()
        # Filters double as the picture shrinks: early blocks need a few simple
        # detectors (edges, dots); deep blocks combine them into many more kinds.
        self.blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        # While training, each of the 256 detector values is zeroed with
        # probability `dropout` (a different random set every step), so no
        # answer can hang on a fragile combination of a few detectors. Off in
        # eval mode. dropout=0 gives back the plain model.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, n_classes)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 256, freq, time) -> (batch, 256): how much each detector fired.

        Average over frequency, then max + average over time. Max keeps a single
        click in 5 s of silence at full strength (the baseline's weak spot);
        average suits steady sounds like rain. Pooling also makes the model work
        for any clip length, which the live demo needs.
        """
        x = x.mean(dim=2)
        return x.amax(dim=2) + x.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.pool(self.blocks(x))))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Scores -> probabilities that sum to 1 (single-label: classes compete)."""
        return torch.softmax(self(x), dim=1)


def shape_check() -> None:
    """Push one real ESC-50 clip through the untrained model and print the size
    after every step. Nothing is trained, so the scores are meaningless; the
    point is that the sizes match the plan before we spend time training."""
    from src.datasets import esc50_logmels

    X, y, fold, meta = esc50_logmels()
    names = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    model = SoundCNN(n_classes=len(names)).eval()

    # One clip -> (batch=1, channel=1, 64, 501): the model always expects a batch.
    x = torch.from_numpy(X[0]).unsqueeze(0).unsqueeze(0)
    print(f"clip 0: {meta['category'][0]!r}, fold {fold[0]}")
    print(f"{'input':15s} {tuple(x.shape)}")
    with torch.no_grad():
        for i, block in enumerate(model.blocks, start=1):
            x = block(x)
            print(f"{f'after block {i}':15s} {tuple(x.shape)}")
        x = model.pool(x)
        print(f"{'global pooling':15s} {tuple(x.shape)}")
        x = model.classifier(x)
        print(f"{'final layer':15s} {tuple(x.shape)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nweights to learn: {total:,}")
    for name, part in [*((f"block {i}", b) for i, b in enumerate(model.blocks, 1)),
                       ("final layer", model.classifier)]:
        print(f"  {name:12s} {sum(p.numel() for p in part.parameters()):>9,}")


if __name__ == "__main__":
    shape_check()

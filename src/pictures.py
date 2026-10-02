"""Log-mel pictures as PNG images, for the website.

Every picture is drawn on the SAME fixed colour scale (DB_RANGE). If each one
were stretched to its own quietest..loudest, a faint recording would look as
bright as a loud one, and two pictures side by side could not be compared.
-100 dB is digital silence (the floor of `audio.log_mel`); training clips
almost never go above +10 dB.
"""

import io

import librosa.display
import numpy as np
from matplotlib.figure import Figure

from src import audio

DB_RANGE = (-100.0, 10.0)
BACKGROUND = "#0c0c0c"  # the site's dark cards
INK = "#a0a0a0"


def log_mel_png(db: np.ndarray, title: str = "") -> bytes:
    """(N_MELS, frames) log-mel in dB -> PNG bytes, with time and Hz axes.

    Uses a bare `Figure` (no pyplot), so several requests can draw at once.
    """
    fig = Figure(figsize=(6.0, 2.8), dpi=110, facecolor=BACKGROUND)
    ax = fig.add_subplot(facecolor=BACKGROUND)
    img = librosa.display.specshow(db, sr=audio.SAMPLE_RATE, hop_length=audio.HOP_LENGTH,
                                   x_axis="time", y_axis="mel", cmap="magma",
                                   vmin=DB_RANGE[0], vmax=DB_RANGE[1], ax=ax)
    bar = fig.colorbar(img, ax=ax, format="%+.0f dB", pad=0.02)
    for a in (ax, bar.ax):
        a.tick_params(colors=INK, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(INK)
    ax.set_xlabel("time (s)", color=INK, fontsize=8)
    ax.set_ylabel("Hz (mel scale)", color=INK, fontsize=8)
    if title:
        ax.set_title(title, color="#ebebeb", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BACKGROUND)
    return buf.getvalue()

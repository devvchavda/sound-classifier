"""Audio loading and features -- the single source of truth.

Training, evaluation, the notebooks and the web server all turn audio into
numbers through THIS file. If the demo computed features even slightly
differently from training (another sample rate, another mel size), the model
would quietly get worse, with no error to tell us why.
"""

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000  # samples per second. 16 kHz keeps everything up to 8 kHz,
                      # which covers speech and most everyday sounds; 44.1 kHz
                      # would triple the data for detail the model barely uses.
N_FFT = 400           # 25 ms analysis window (400 / 16000)
HOP_LENGTH = 160      # 10 ms step between windows -> 100 frames per second
N_MELS = 64           # 64 mel bands: frequency axis of our "sound picture"
N_MFCC = 20


def load(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any audio file -> mono float32 waveform at `sr` Hz, values in [-1, 1]."""
    wave, _ = librosa.load(path, sr=sr, mono=True)
    return wave.astype(np.float32)


def log_mel(wave: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Waveform -> log-mel spectrogram, shape (N_MELS, frames).

    1. STFT: slide a 25 ms window along the audio, and for each position ask
       "how much energy at each frequency?" (a Fourier transform).
    2. Mel: merge those ~200 frequency bins into 64 bands spaced the way ears
       hear -- fine detail at low pitches, coarse at high ones.
    3. Log (decibels): loudness is perceived logarithmically; without it a loud
       bang would dwarf everything else in the picture.
    """
    mel = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


def mfcc(wave: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Waveform -> MFCCs, shape (N_MFCC, frames): a compressed summary of the
    log-mel's *shape* (classic hand-crafted feature, used by the baseline)."""
    return librosa.feature.mfcc(
        y=wave, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    ).astype(np.float32)

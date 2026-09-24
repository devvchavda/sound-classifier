# Project journal

A running log of things worth putting in the report/PPT: results, figures, decisions and why, and
failures and their fixes. The newest entry is at the bottom.

## 2026-09-24: Day 1, setup

- **Problem framing:** multi-label, not multi-class. A clip can contain a dog AND speech, and each
  sound gets its own independent probability. → sigmoid per class, not softmax.
- **Data:** ESC-50 (2000 × 5 s clips, 50 classes, 5 official folds) plus a "speech" class cut from
  Mini LibriSpeech. Speech is added through `data/custom/speech/`, the same route any future class
  (e.g. fan) takes.
- **Leakage rule:** always split by fold. ESC-50 keeps clips from one recording in one fold. For
  speech, one speaker stays in one fold, so the model is never tested on a voice it trained on.
- **Feature choice:** 16 kHz mono → 64-band log-mel spectrogram (25 ms window, 10 ms hop). 16 kHz
  keeps everything up to 8 kHz, which is plenty for speech and everyday sounds, at about 1/3 the
  data of 44.1 kHz.

## 2026-09-24: Day 1, baseline results (MFCC stats → sklearn, 5-fold CV on ESC-50)

| Model | Accuracy (mean ± std over 5 folds) |
|---|---|
| Chance (1 of 50) | 0.020 |
| Logistic Regression | 0.478 ± 0.029 |
| Random Forest (500 trees) | **0.541 ± 0.023** |

- Figures: `reports/figures/baseline_*_confusion.png`
- **Where the baseline does well:** steady, texture-like sounds such as thunderstorm, rain,
  toilet flush, crying baby and rooster. Their average over 5 s is a good fingerprint.
- **Where it fails:** short, transient events such as door wood creaks, clock tick, mouse click,
  water drops, breathing (confused with crow) and car horn. Many ESC-50 clips are a short event
  padded with silence, and averaging over 5 s mixes the event with the silence and erases the
  timing. **That is the argument for the CNN:** it looks at *where* and *when* the energy is in the
  spectrogram, not just its average.
- Features take ~1.5 min to compute once; they are cached in `data/cache/`.
- Speech class ready: 130 × 5 s clips from 26 LibriSpeech speakers in `data/custom/speech/`
  (Mini LibriSpeech has fewer speakers than expected; we cap at 5 clips per speaker).

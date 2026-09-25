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

## 2026-09-24: Reading spectrograms (from exploring ESC-50)

- **Four visual building blocks:** horizontal line = steady pitch; stacked parallel lines =
  something with a voice/pitch (harmonics: rooster, cat, baby, siren); vertical line = click/hit
  (clock tick, knock, keyboard); fog = noise (rain, wind, helicopter, sneeze).
- **Some classes differ only in rhythm over time:** clock tick (regular), door knock (grouped) and
  keyboard (random) are all vertical lines. Averaging over time throws that away, which fits the
  baseline's failures on transient sounds.
- **Padded silence:** short ESC-50 clips are padded with exact zeros (pure black in the
  spectrogram). Real audio is never that silent. Watch that the CNN doesn't learn "lots of black
  = short-event class".
- **Open-set limitation (for the demo):** the model can only answer with classes it was trained
  on. An unknown noise like a fan or AC hum will likely be tagged as the nearest fog-like class
  (rain). Possible fix later: a "background noise" class in `data/custom/`.

## 2026-09-25: CNN design (each choice discussed before writing)

- **Single-label first** (softmax over 50 classes), so the CNN answers the same question on the
  same folds as the baseline's 54%. Multi-label later only swaps softmax for sigmoid and the loss.
- **Input normalisation:** one global mean and std, measured on the training folds only (the test
  fold would leak), saved with the model so the demo shifts live audio by exactly the same amount.
  Per-clip normalisation was rejected: it makes quiet and loud clips look identical.
- **Log-mel pictures cached once** (2000 × 64 × 501, ~250 MB, ~30 s to build), because training
  shows every clip dozens of times.
- **Architecture:** 4 blocks of 3×3 conv → BatchNorm → ReLU → MaxPool 2×2, with filters
  32 → 64 → 128 → 256. The picture shrinks 64×501 → 4×31.
- **Global pooling instead of flattening:** mean over frequency, then max + mean over time →
  256 numbers. Flattening would cost 1.6 M weights in the last layer alone, tie detection to
  position, and only work for exactly 5 s input. Max keeps a lone click at full strength (the
  baseline's weak spot); mean suits steady sounds.
- **Size:** 401,650 weights (block 4 alone is 295,680), versus ~1,600 training clips, so
  overfitting is expected.
- **First run deliberately without dropout or augmentation**, so the overfitting gap is visible
  in the curves before the fixes are added (before/after figure for the report).
- Shape check (`python -m src.models.cnn`): one real clip goes through the untrained model, and the
  size after each block matches the design.

## 2026-09-25: Training setup

- Loss: cross-entropy (accuracy is flat to small weight nudges, so it gives no slope to follow).
  Sanity check: an untrained model should score about −log(1/50) = 3.9.
- Adam, learning rate 0.001, cosine decay to ~0 by the last epoch. Batches of 32, reshuffled every
  epoch, for 60 epochs. Normalisation from the training folds: mean −43.0 dB, std 27.8 dB (test fold 5).
- **Reporting rule:** report the last epoch's test accuracy, fixed in advance. Choosing the best
  epoch by its test score would let the test fold pick the model.

## 2026-09-25: First CNN run, `no_dropout` (no dropout, no augmentation), test fold 5

- **Test accuracy 0.790** (last epoch, per the reporting rule), against the baseline's 0.541
  (5-fold mean; the CNN's own 5-fold number comes later). 60 epochs in 3.3 min on the RTX 4060.
- Untrained test loss 3.915 against a blind guess of 3.912, so the wiring is right.
- **Overfitting, clearly visible:** train accuracy 1.000 and train loss 0.009 (training set
  memorised), while test loss stays near 0.80. That 21-point train/test gap is what dropout and
  augmentation have to shrink.
- The test loss did not turn back up (no U-shape). It fell until ~epoch 30, then went flat at ~0.8.
  Once the train loss is ~0 there is no slope left and the lr is decaying, so the model stops
  changing: epochs ~27–60 taught it nothing. Here overfitting shows as "stuck", not "getting
  worse". The spike at epoch 25 is in the train loss too, so it was one oversized step, not
  the test data.
- Early test accuracy wobbles hard (epoch 9 0.642 → epoch 10 0.555 → epoch 12 0.718): big steps
  at lr 0.001, and each of the 400 test clips is 0.25 %. The last 10 epochs sit at 0.782–0.792
  as the cosine schedule shrinks the steps, which is the settling Decision 7 was for.
- Figure: `runs/no_dropout/curves.png`.

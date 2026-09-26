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

## 2026-09-25: Run 2, `with_dropout` (dropout 0.5 before the final layer), test fold 5

- **Test accuracy 0.790 (last epoch), exactly the same as without dropout.** No measurable gain.
  Peak was 0.805 at epoch 59, not reportable by the rule; 0.805 → 0.790 is 6 clips out of 400.
- The blue train curve looked much less overfit (0.921 acc, 0.29 loss), but that was measured with
  dropout switched ON. Re-measured with dropout off: **train 0.993**, test 0.790. The model still
  memorises the training set almost perfectly. The gap is still ~20 points.
- Why: dropout only disrupts the last layer. The conv blocks (388k of the 401k weights) still see
  the exact same 1,600 pictures every epoch and can memorise them. That points to augmentation:
  change the pictures every epoch so there is nothing exact to memorise.
- Early test accuracy wobbled even more (epoch 26 dropped to ~0.55); training was slower
  (test 0.75 at epoch 40 vs 0.78 at epoch 30 without dropout).

## 2026-09-25: Web demo v1 (upload audio/video or YouTube link → probabilities)

- **Sliding window:** 5 s windows (the training length), one every 1 s. Overall score = each
  class's highest probability in any window; timeline = winning class per window.
- **Same path as training, verified:** ffmpeg only unpacks the audio, and `audio.load` does the
  resampling as in training (identical samples on an ESC-50 clip, max difference 0.0). Running the
  web path over all of fold 5 gives 316/400 = **0.790**, the same number the training script reported.
- **Test video** (25 s mp4 of five fold-5 clips: dog, rain, clock tick, rooster, siren): all five
  are the top 5 overall. Windows that straddle two clips produce wrong labels (crow, footsteps,
  cow): a mix of two sounds is something the single-label model never saw. That is the argument
  for the multi-label stage.
- **Open-set in practice:** "Me at the zoo" (YouTube, a man talking) → sheep 0.58. Speech is not
  one of the 50 classes, and softmax has to pick one of them.

## 2026-09-25: Web demo v2 (play it and watch the labels change)

- **"Live" labels without live inference:** the server analyses the whole file once and now also
  returns every 5 s slice's full prediction with its centre time. While the video/audio plays,
  the page picks the slice nearest to `currentTime` and shows its top 5. Same model, same
  numbers as the timeline, no GPU work during playback. This is the plan's "pre-recorded video"
  mode; the camera/mic mode still needs real streaming.
- **YouTube first, Shorts included:** the video is embedded with YouTube's player; only the audio
  is downloaded for the model. A `/shorts/` link is not always a tall video (some Shorts are
  uploaded landscape), so the player's shape comes from the video's real width and height.
- **Layout:** one top-to-bottom flow (link → player + "now hearing" → timeline → whole-clip
  summary → the list of the 50 sounds it knows, grouped by ESC-50's five categories). Before, the
  answer appeared top-right while the input sat bottom-left.
- **Bug found while testing:** the probability bars had always been empty. The coloured fill
  was a `<span>`, and inline elements ignore `width`; `display: block` fixed it.

## 2026-09-26: SpecAugment (run 3: worse, 0.752 vs 0.790)

- **What:** every training batch gets 2 random pitch stripes (up to 8 of 64 mel bands, muted for
  the whole clip) and 2 random time stripes (up to 0.5 s, all pitches muted). About 20% of each
  picture is hidden, a different 20% every time. Test clips are never masked. Same fold 5, seed,
  60 epochs and dropout 0.5 as run 2; the stripes use their own random stream, so the shuffle
  order is identical and the masks are the only difference.
- **Why:** run 2 showed memorisation (train 0.993 vs test 0.790). Hiding a different part of
  each picture every epoch should leave nothing exact to memorise. Time shift was skipped: the
  CNN's global pooling already mostly ignores *where* a sound is, so it was expected to do little.
- **Result: test 0.752 (last epoch), down from 0.790.** Clean train accuracy fell to 0.946 (from
  0.993), so it did memorise less, but the test score fell by about as much: the train-test gap
  stayed ~19-20 points. Test accuracy was flat around 0.75 for the last 15 epochs.
- **Reading:** the stripes made the task harder without making the model generalise better.
  Suspects: too strong on top of dropout, or 60 epochs too few for a harder task. Not yet tested.

## 2026-09-26: Smaller stripes (run 4: 0.770), SpecAugment dropped

- Same as run 3 but 1 pitch + 1 time stripe per clip (~10% hidden instead of ~20%).
- **Test 0.770, clean train 0.975.** The trend is one-directional:

  | stripes per kind | clean train | test (last epoch) | gap |
  |---|---|---|---|
  | 0 (run 2) | 0.993 | **0.790** | 20 pts |
  | 1 (run 4) | 0.975 | 0.770 | 20 pts |
  | 2 (run 3) | 0.946 | 0.752 | 19 pts |

- More masking lowers both numbers together; the gap never closes. So on this model and data,
  masking does not help generalisation, it only makes the task harder. Differences of 0.02 are
  8 clips out of 400 and could partly be run-to-run noise, but the direction is the same at both
  sizes.
- The student suspected the learning rate. Checked against the logs: clean train accuracy stays
  high (0.95-0.99), so the model learns fine; the problem is the gap, which a learning-rate change
  moves up or down together rather than closing. Not tested with a run.
- **Decision:** SpecAugment stays in the code (`--specaugment N`, default off) but is not used.
  `runs/with_dropout` (0.790) remains the model.

## 2026-09-26: Random volume (run 5: 0.787, no change)

- **What:** every training clip is played at a random volume between -10 and +10 dB (10 dB is
  roughly what people hear as "twice / half as loud"), new each batch. Done on the cached log-mel:
  in decibels, louder by g dB = +g on every cell. Digital silence (-100 dB, 10% of all cells, from
  padded clips) is left untouched, as real audio would. Checked against actually rescaling the
  audio and recomputing: identical at -10 dB; at +10 dB only 36 of 32,064 cells differ, all at
  -90 to -100 dB (inaudible), mean difference 0.007 dB.
- **Why:** YouTube and phone audio come at very different volumes; ESC-50's test clips don't.
- **Result: test 0.787 vs 0.790 (1 clip out of 400), clean train 0.991.** No effect on fold 5
  either way, and the train-test gap is unchanged.
- **Reading:** fold 5 can't show what gain is for. Its clips are at ESC-50's own recording level,
  so a model that falls apart on quiet or loud audio would still score the same here.
- **Follow-up test: fold 5 played at other volumes** (no retraining; same silence rule as training):

  | volume change | -20 dB | -10 dB | 0 dB | +10 dB | +20 dB |
  |---|---|---|---|---|---|
  | no gain (run 2) | 0.522 | 0.740 | **0.790** | 0.725 | 0.570 |
  | gain +/-10 dB (run 5) | **0.685** | **0.780** | 0.788 | **0.775** | **0.728** |

  At the recorded volume they tie. Away from it the old model collapses (loses 27 points at
  -20 dB); the gain model loses 10 at worst, and holds up even at +/-20 dB, beyond the +/-10 it was
  trained with. The plain fold-5 score hid this completely: **a test set only measures the
  conditions it contains.**
- **Decision:** keep gain. Run 5 (`runs/gain10`) is the better model for real-world audio.

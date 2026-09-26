# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A college ML project (sem 7). It is a **multi-label sound tagger**: give it any audio and it returns an
independent probability per sound source (dog 0.7, speech 0.8, keyboard 0.02). The probabilities do
not sum to 1, so the final model uses **sigmoid + BCE, not softmax**. The web demo (later) has three
modes: upload audio, a live camera/mic session streamed over WebSocket, and a pre-recorded video
whose timeline is analyzed once on the server and synced to `video.currentTime` in the browser.

The full phased plan (day-by-day, repo layout, verification) lives in
`C:\Users\devch\.claude\plans\enchanted-plotting-rivest.md`.

## How to work with this student

- They are new to ML and learn by building. Before each piece of code, explain *what* and *why* in
  a few lines, and introduce concepts at the moment they're needed, not up front.
- Expect "why not something simpler?" questions and answer them with evidence. The MFCC baseline
  exists for exactly this reason.
- Stop at checkpoints so they can run things themselves. Don't race through several days' work.
- Build from scratch where possible. The pretrained model (PANNs CNN14) is for comparison and
  **inference only. Never fine-tune it.** That was the student's explicit rule, so the comparison
  stays honest.
- Don't write the report or PPT. Add a dated entry to `reports/journal.md` whenever something is
  presentation-worthy (a result, a figure, a decision and its reason, a bug and its fix).

## Invariants

- **All audio → numbers goes through `src/audio.py`** (16 kHz mono, 64-band log-mel, 10 ms hop).
  Training, evaluation, notebooks and the server must share it; a mismatch silently degrades the demo.
- **Split by fold, never by random shuffle.** For ESC-50, use its `fold` column (clips from one
  source recording share a fold). For `data/custom/<label>/`, files are named
  `<source>__<rest>.wav`, and every clip with the same `<source>` goes in the same fold (for speech,
  `<source>` is the LibriSpeech speaker).
- **Classes are data-driven, not hardcoded.** Any folder `data/custom/<label>/` becomes a class. A
  saved checkpoint must store its own label list.
- Code lives in `src/`. Notebooks stay thin: they import from `src/` and show plots.

## Commands

Windows, Python 3.13, RTX 4060 (CUDA). Run everything from the repo root with the venv's python:

```
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/download_data.py      # ESC-50 + speech clips -> data/ (gitignored)
.venv/Scripts/python -m src.baseline               # MFCC + sklearn, 5-fold CV, figures -> reports/figures/
.venv/Scripts/python -m src.models.cnn             # CNN shape check: one clip through the untrained model
.venv/Scripts/python -m src.train --fold 5 --epochs 60 --gain 10 --name gain10   # train (current best); run folder -> runs/
.venv/Scripts/python -m src.train_multilabel --fold 5 --epochs 150 --name multilabel_150   # multi-label (51 sounds, mixes)
.venv/Scripts/python -m src.train_multilabel --fold 5 --val-fold 4 --epochs 150 --name multilabel_val4   # + fold 4 as practice exam
.venv/Scripts/python -m src.calibrate multilabel_val4                              # honesty check, correction, cut-off (on fold 4)
.venv/Scripts/python -m src.calibrate multilabel_150 --from multilabel_val4        # reuse them on the 4-fold model
.venv/Scripts/python -m src.panns_compare     # our CNN vs PANNs CNN14 (inference only, label matching), fold 5
.venv/Scripts/python -m src.infer some_file.mp4    # sliding-window prediction on any audio/video file
.venv/Scripts/python -m uvicorn server.app:app --port 8000   # web demo on http://localhost:8000, live page /live.html (SOUND_RUN=<run> picks the model)
.venv/Scripts/python -m jupyter notebook notebooks/ # learning notebooks (01_explore: sound -> log-mel)
```

Baseline MFCC features are cached in `data/cache/baseline_features.npz`, and the CNN's log-mel
pictures in `data/cache/esc50_logmel.npz` (+ `custom_logmel.npz` for `data/custom/`). Delete them after changing `src/audio.py`, or training
will keep using stale features.

PANNs needs `data/panns/Cnn14_16k_mAP=0.438.pth` (Zenodo record 3987831) and its label list at
`~/panns_data/class_labels_indices.csv`: the `panns_inference` package reads it from there and tries to
fetch it with `wget`, which Windows lacks, so download it by hand.

ffmpeg is not installed system-wide. `src/audio.load_any` uses the `imageio-ffmpeg` pip package to unpack
any audio/video file; resampling still goes through `load` (librosa), exactly as for training.

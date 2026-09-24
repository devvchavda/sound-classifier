"""Download the datasets into data/.

    python scripts/download_data.py

- ESC-50  -> data/ESC-50-master/   (2000 clips, 50 classes, 5 folds, ~600 MB)
- Speech  -> data/custom/speech/   (5 s clips cut from Mini LibriSpeech dev-clean-2)

Speech goes into data/custom/<label>/ on purpose: that folder is how ANY new
class gets added (e.g. data/custom/fan/). File names follow the rule
"<source>__<anything>.wav": clips that share a <source> (here: the speaker) are
always kept in the same fold, so the model is never tested on a voice it
trained on. Same idea as ESC-50 keeping clips from one recording in one fold.
"""

import io
import sys
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
LIBRISPEECH_URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"

CLIP_SECONDS = 5
MAX_CLIPS_PER_SPEAKER = 5  # ~40 speakers -> ~200 clips; avoids one voice dominating


def download(url: str) -> bytes:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        done = 0
        while chunk := resp.read(1 << 20):
            buf.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    return buf.getvalue()


def get_esc50() -> None:
    target = DATA / "ESC-50-master"
    if (target / "meta" / "esc50.csv").exists():
        print("ESC-50 already present, skipping.")
        return
    with zipfile.ZipFile(io.BytesIO(download(ESC50_URL))) as zf:
        zf.extractall(DATA)
    print(f"ESC-50 -> {target}")


def get_speech() -> None:
    out_dir = DATA / "custom" / "speech"
    if out_dir.exists() and any(out_dir.glob("*.wav")):
        print("Speech clips already present, skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    clips_per_speaker: dict[str, int] = defaultdict(int)
    written = 0
    with tarfile.open(fileobj=io.BytesIO(download(LIBRISPEECH_URL)), mode="r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".flac"):
                continue
            # LibriSpeech file name: <speaker>-<chapter>-<utterance>.flac
            utt_id = Path(member.name).stem
            speaker = utt_id.split("-")[0]
            if clips_per_speaker[speaker] >= MAX_CLIPS_PER_SPEAKER:
                continue
            audio, sr = sf.read(io.BytesIO(tf.extractfile(member).read()), dtype="float32")
            n = CLIP_SECONDS * sr
            if len(audio) < n:  # skip utterances shorter than one clip
                continue
            # take the middle 5 s: avoids the silence at the start/end of a sentence
            start = (len(audio) - n) // 2
            clip = audio[start : start + n]
            sf.write(out_dir / f"{speaker}__{utt_id}.wav", clip, sr)
            clips_per_speaker[speaker] += 1
            written += 1

    print(f"Speech: {written} clips from {len(clips_per_speaker)} speakers -> {out_dir}")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    get_esc50()
    get_speech()
    sys.exit(0)

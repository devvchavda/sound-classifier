"""Web demo: upload audio/video or paste a YouTube link -> sound probabilities,
plus a live page (camera + microphone, /live.html) fed over a WebSocket.

    .venv/Scripts/python -m uvicorn server.app:app --port 8000
    then open http://localhost:8000

The model is loaded once at startup and stays on the GPU. Pick another trained
run with the SOUND_RUN environment variable (default: multilabel_150). Scores are
corrected with the run's calibration.json, and the page gets the two cut-offs
(heard / maybe) with every answer.
Every answer also compares, for each sound found, the 5 s slice where it scored
highest with the training clip the model finds most like it (src/similar.py).
That needs data/; without it the site works, minus the comparison.
Each sound also has its own page (sound.html?label=dog), fed by /api/sound/<label>.
Endpoints are plain `def`, so FastAPI runs them in a worker thread and the slow
parts (ffmpeg, downloads, the model) don't block the server.
"""

import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import audio
from src.datasets import ROOT, esc50_meta
from src.infer import HEARD, WINDOW_S, Tagger
from src.train import RUNS
from src.similar import TrainingClips

MAX_SECONDS = 600  # 10 min: longer uploads are cut, longer YouTube videos refused

# ESC-50 numbers its classes so that target // 10 is one of five broad groups.
ESC50_GROUPS = ["Animals", "Nature and water", "Human sounds", "Around the house", "Outdoors and city"]

tagger = Tagger(os.environ.get("SOUND_RUN", "multilabel_150"))
try:
    clips = TrainingClips(tagger)
except (FileNotFoundError, ValueError) as e:  # no data/ yet, or data/ changed since training
    print(f"No training-clip comparison: {e}")
    clips = None
app = FastAPI(title="Sound tagger")


def _label_groups(labels: list[str]) -> list[dict]:
    """The model's own labels, grouped for display. A label that is not an ESC-50
    class (a data/custom/<label>/ folder, e.g. speech) goes under "Added by us"."""
    target = dict(esc50_meta()[["category", "target"]].drop_duplicates().values)
    groups = {name: [] for name in ESC50_GROUPS + ["Added by us"]}
    for label in labels:
        name = ESC50_GROUPS[target[label] // 10] if label in target else "Added by us"
        groups[name].append(label)
    return [{"name": name, "labels": sorted(ls)} for name, ls in groups.items() if ls]


GROUPS = _label_groups(tagger.labels)
GROUP_OF = {label: g["name"] for g in GROUPS for label in g["labels"]}
ORDER = [label for g in GROUPS for label in g["labels"]]  # page order, for previous / next


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# Per-sound test results, if they exist. AP: on fold 5's fixed test mixes
# (src/train_multilabel.py). PANNs: share of a sound's single fold-5 clips
# named right, by our model and by PANNs (src/panns_compare.py), only shown
# when that comparison was made with the model the site runs.
AP = _json(RUNS / tagger.run / "per_class_ap.json") or {}
_panns = _json(RUNS / "panns_compare.json") or {}
_ours = [k for k in _panns.get("top1", {}) if k.startswith("our CNN")]
PANNS = _panns.get("per_sound", {}) if _ours and _ours[-1] == f"our CNN ({tagger.run})" else {}


@app.get("/api/info")
def info():
    return {"model": tagger.run, "device": tagger.device.type, "classes": len(tagger.labels),
            "labels": tagger.labels, "groups": GROUPS, "max_seconds": MAX_SECONDS,
            "multilabel": tagger.multilabel, "calibrated": tagger.calibration is not None,
            "heard_cut": HEARD, "maybe_cut": round(tagger.maybe_cut, 3)}


@app.post("/api/predict")
def predict_file(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{Path(file.filename or '').suffix}"
        with path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        result = _analyze(path)
    return {"source": file.filename, **result}


class Link(BaseModel):
    url: str


@app.post("/api/predict-url")
def predict_url(link: Link):
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    with tempfile.TemporaryDirectory() as tmp:
        # Only the audio track, and no playlists: one link = one video.
        opts = {"format": "bestaudio/best", "outtmpl": f"{tmp}/audio.%(ext)s",
                "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True}
        try:
            with YoutubeDL(opts) as ydl:
                meta = ydl.extract_info(link.url, download=False)
                if (meta.get("duration") or 0) > MAX_SECONDS:
                    raise HTTPException(400, f"Video is {meta['duration'] // 60} min long; "
                                             f"the limit is {MAX_SECONDS // 60} min.")
                ydl.process_ie_result(meta, download=True)
        except DownloadError as e:
            raise HTTPException(400, f"Could not download that link. If it is a normal "
                                     f"video, try `pip install -U yt-dlp`. ({e})")
        files = list(Path(tmp).glob("audio.*"))
        if not files:
            raise HTTPException(400, "The download produced no audio file.")
        result = _analyze(files[0])
    return {"source": meta.get("title") or link.url, "video_id": meta.get("id"),
            "vertical": _is_vertical(meta, link.url), **result}


def _is_vertical(meta: dict, url: str) -> bool:
    """Shorts are 9:16, so the page gives them a tall player instead of a wide one."""
    sizes = [(f["width"], f["height"]) for f in meta.get("formats") or []
             if f.get("width") and f.get("height")]
    if sizes:
        w, h = max(sizes, key=lambda wh: wh[1])
        return h > w
    return "/shorts/" in url


def _analyze(path: Path) -> dict:
    try:
        wave = audio.load_any(path, max_seconds=MAX_SECONDS)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if len(wave) == 0:
        raise HTTPException(400, "That file has no audio in it.")
    result = tagger.predict(wave)
    result["compare"] = clips.compare(wave, result) if clips else []
    result["truncated"] = result["duration"] >= MAX_SECONDS - 0.5  # only the start was used
    return result


@app.get("/api/sound/{label}")
def sound(label: str):
    if label not in tagger.labels:
        raise HTTPException(404, f"The model doesn't know a sound called {label!r}.")
    k = ORDER.index(label)
    ranked = sorted(AP, key=AP.get, reverse=True)
    return {
        "label": label,
        "group": GROUP_OF[label],
        "esc50": GROUP_OF[label] != "Added by us",
        "counts": clips.counts[label] if clips else None,
        "test_fold": tagger.config["fold"],
        "practice_fold": tagger.config.get("val_fold"),
        "ap": AP.get(label),
        "ap_rank": ranked.index(label) + 1 if label in AP else None,
        "ap_of": len(ranked),
        "panns": PANNS.get(label),
        "examples": [{"id": i, "name": clips.name(i)} for i in clips.typical(label)] if clips else [],
        "previous": ORDER[k - 1],
        "next": ORDER[(k + 1) % len(ORDER)],
        "model": tagger.run,
    }


@app.get("/api/clip/{i}.png")
def clip_png(i: int):
    _check_clip(i)
    return Response(clips.png(i), media_type="image/png",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/api/clip/{i}.wav")
def clip_wav(i: int):
    _check_clip(i)
    buf = io.BytesIO()
    soundfile.write(buf, clips.wav(i), audio.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"Cache-Control": "max-age=86400"})


def _check_clip(i: int):
    if clips is None or not 0 <= i < len(clips.paths):
        raise HTTPException(404, "No such training clip.")


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """Live page: the browser sends raw microphone audio every 0.5 s; each time we
    answer with the scores of the last 5 s (as long as a training clip).

    First message: {"sample_rate": 48000} (the microphone's own rate). Then only
    binary messages of float32 samples. The audio is kept in memory, only the
    last 5 s, and never saved. Resampling to 16 kHz happens here, with the same
    resampler as training (src/audio.py), not in the browser.
    """
    await ws.accept()
    try:
        sr = int((await ws.receive_json())["sample_rate"])
        keep = int(WINDOW_S * sr)
        heard = np.zeros(0, dtype=np.float32)
        while True:
            chunk = np.frombuffer(await ws.receive_bytes(), dtype=np.float32)
            heard = np.concatenate([heard, chunk])[-keep:]
            probs = await run_in_threadpool(lambda: tagger.score_window(audio.resample(heard, sr)))
            await ws.send_json({"probs": np.round(probs, 3).tolist(),
                                "filled": round(len(heard) / sr, 1)})
    except WebSocketDisconnect:
        pass


# Last, so /api/* routes win over the static files.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

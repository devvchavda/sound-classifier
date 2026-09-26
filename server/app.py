"""Web demo: upload audio/video or paste a YouTube link -> sound probabilities.

    .venv/Scripts/python -m uvicorn server.app:app --port 8000
    then open http://localhost:8000

The model is loaded once at startup and stays on the GPU. Pick another trained
run with the SOUND_RUN environment variable (default: multilabel_150). Scores are
corrected with the run's calibration.json, and the page gets the two cut-offs
(heard / maybe) with every answer.
Endpoints are plain `def`, so FastAPI runs them in a worker thread and the slow
parts (ffmpeg, downloads, the model) don't block the server.
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import audio
from src.datasets import ROOT, esc50_meta
from src.infer import HEARD, Tagger

MAX_SECONDS = 600  # 10 min: longer uploads are cut, longer YouTube videos refused

# ESC-50 numbers its classes so that target // 10 is one of five broad groups.
ESC50_GROUPS = ["Animals", "Nature and water", "Human sounds", "Around the house", "Outdoors and city"]

tagger = Tagger(os.environ.get("SOUND_RUN", "multilabel_150"))
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
    result["truncated"] = result["duration"] >= MAX_SECONDS - 0.5  # only the start was used
    return result


# Last, so /api/* routes win over the static files.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

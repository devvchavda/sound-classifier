"""Start the website in one step and open it in the browser.

    .venv/Scripts/python -m server                 (or double-click run.bat)
    .venv/Scripts/python -m server --run multilabel_val4 --port 8001 --no-browser

Before loading anything heavy it checks the two things the site can't run
without, and says which command fixes each, instead of a long traceback.
"""

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

from src.datasets import ROOT

RUNS = ROOT / "runs"  # same folder as src.train.RUNS, without importing torch yet


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the sound tagger website.")
    parser.add_argument("--run", default=os.environ.get("SOUND_RUN", "multilabel_150"),
                        help="trained model, a folder under runs/ (default: multilabel_150)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    if not (RUNS / args.run / "model.pt").exists():
        trained = sorted(p.parent.name for p in RUNS.glob("*/model.pt"))
        print(f"No trained model runs/{args.run}/model.pt.")
        if trained:
            print(f"Trained runs here: {', '.join(trained)}. Pick one with --run <name>.")
        else:
            print("Train one first (about an hour on the GPU):\n"
                  "  .venv/Scripts/python -m src.train_multilabel --fold 5 --epochs 150 --name multilabel_150\n"
                  "  .venv/Scripts/python -m src.calibrate multilabel_150 --from multilabel_val4")
        sys.exit(1)
    if not (ROOT / "data" / "ESC-50-master").exists():
        print("No data/ESC-50-master: the site will work, but without spectrogram comparisons\n"
              "and example clips. To get them: .venv/Scripts/python scripts/download_data.py\n")

    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"Port {args.port} is already in use. Is the site already running? Open "
                  f"http://localhost:{args.port}, or start another one with --port {args.port + 1}.")
            sys.exit(1)

    url = f"http://localhost:{args.port}"
    print(f"Loading the model ({args.run}) and the training clips...")
    os.environ["SOUND_RUN"] = args.run
    import uvicorn

    from server.app import app  # loads the model: takes a few seconds

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    print(f"\nWebsite: {url}   (live page: {url}/live.html)   Stop it with Ctrl+C.\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def open_when_ready(url: str) -> None:
    """Open the browser once the server answers, not before (it would show an error page)."""
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{url}/api/info", timeout=1)
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.5)


if __name__ == "__main__":
    main()

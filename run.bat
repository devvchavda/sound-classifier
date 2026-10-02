@echo off
rem Double-click to start the website; it opens in your browser by itself.
rem Extra options go through, e.g.:  run.bat --run multilabel_val4 --port 8001
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No .venv here yet. One-time setup, in this folder:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  echo   .venv\Scripts\python scripts\download_data.py
  pause
  exit /b 1
)

.venv\Scripts\python -m server %*
pause

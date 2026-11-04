#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  for python in python3.12 python3.13 python3.11 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12; do
    if command -v "$python" >/dev/null 2>&1; then
      "$python" -m venv .venv && break
    fi
  done
fi
if [ ! -x .venv/bin/python ]; then
  echo "Se necesita Python 3.11, 3.12 o 3.13 (python.org/downloads)."
  exit 1
fi
if ! .venv/bin/python -c 'import PySide6, pyvista, scipy, matplotlib, reportlab' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
.venv/bin/python main.py

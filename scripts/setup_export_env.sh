#!/usr/bin/env bash
# Minimal Python env ONLY for FireNet .pth → ONNX export (the one non-C++ step).
# Prefer micromamba/conda/venv — your choice. Example with venv:
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/libeventgate-export}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install onnx onnxscript numpy
# Optional checkpoint helper:
# pip install gdown

echo "Activate: source $VENV/bin/activate"
echo "Then:     python $ROOT/python/export_firenet_onnx.py --help"

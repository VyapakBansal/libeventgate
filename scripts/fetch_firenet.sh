#!/usr/bin/env bash
# Clone official FireNet (PyTorch) for the ONNX export only.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="${ROOT}/third_party"
mkdir -p "${THIRD}"
DEST="${THIRD}/rpg_e2vid"

if [[ -d "${DEST}/.git" ]]; then
  git -C "${DEST}" fetch origin
  git -C "${DEST}" checkout cedric/firenet
  git -C "${DEST}" pull --ff-only origin cedric/firenet || true
else
  git clone --branch cedric/firenet --single-branch \
    https://github.com/cedric-scheerlinck/rpg_e2vid.git "${DEST}"
fi

mkdir -p "${ROOT}/weights" "${ROOT}/engines"

echo "Repo: ${DEST}"
echo ""
echo "Download checkpoint firenet_1000.pth.tar → ${ROOT}/weights/"
echo "  Google Drive: https://drive.google.com/file/d/1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa"
echo "  (browser or gdown: pip install gdown && gdown 1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa -O weights/firenet_1000.pth.tar)"
echo ""
echo "Then export ONNX:"
echo "  bash scripts/setup_export_env.sh   # once"
echo "  source ~/venvs/libeventgate-export/bin/activate"
echo "  python python/export_firenet_onnx.py \\"
echo "    --repo third_party/rpg_e2vid \\"
echo "    --checkpoint weights/firenet_1000.pth.tar \\"
echo "    --out engines/firenet.onnx \\"
echo "    --height 512 --width 640"

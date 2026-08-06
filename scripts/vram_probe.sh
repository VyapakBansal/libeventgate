#!/usr/bin/env bash
# VRAM snapshot helper (you run between steps).
set -eu
LABEL="${1:-probe}"
TS="$(date -Iseconds)"
OUT="${VRAM_LOG:-$HOME/libeventgate_vram.log}"
LINE=$(nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits)
echo "${TS} | ${LABEL} | ${LINE}" | tee -a "${OUT}"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "${USED:-0}" -gt 4500 ] 2>/dev/null; then
  echo "WARN: VRAM used ${USED} MiB > 4500 soft budget" >&2
fi

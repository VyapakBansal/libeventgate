#!/usr/bin/env bash
# Day 2–3 helper: after engines/firenet.engine exists, run proxy/real recon blackout demo.
# You supply EVENTS H5 (and optional IMU). Synthetic smoke stays: ./build/phase0_smoke
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
ENGINE="${ENGINE:-$ROOT/engines/firenet.engine}"
OUT="${OUT:-$ROOT/out/blackout_demo}"
EVENTS="${1:-}"
IMU="${2:-}"

if [[ -z "${EVENTS}" ]]; then
  echo "Usage: $0 events.h5 [imu.csv]"
  echo "  ENGINE=engines/firenet.engine OUT=out/before  optional env overrides"
  exit 2
fi

ARGS=(
  --events "${EVENTS}"
  --out "${OUT}"
  --window-ms 10
  --bins 5
  --blackout-tail-s 2
  --vram-every 50
)
if [[ -f "${ENGINE}" ]]; then
  ARGS+=(--engine "${ENGINE}")
  echo "using engine: ${ENGINE}"
else
  echo "WARN: no engine at ${ENGINE} — proxy polarity recon only"
fi
if [[ -n "${IMU}" ]]; then
  ARGS+=(--imu "${IMU}")
fi

exec "${BUILD}/phase0_run" "${ARGS[@]}"

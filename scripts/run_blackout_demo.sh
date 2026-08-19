#!/usr/bin/env bash
# Reconstruct with a 2 s empty tail after the last event (HOLD past end-of-stream).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
ENGINE="${ENGINE:-$ROOT/engines/firenet.engine}"
OUT="${OUT:-$ROOT/out/blackout_demo}"
EVENTS="${1:-}"
IMU="${2:-}"

if [[ -z "${EVENTS}" ]]; then
  echo "Usage: $0 events.h5 [imu.csv]"
  echo "  ENGINE=engines/firenet.engine OUT=out/recon  optional env overrides"
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

exec "${BUILD}/eventgate" "${ARGS[@]}"

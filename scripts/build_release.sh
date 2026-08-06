#!/usr/bin/env bash
# Configure + build Release with Ninja (you run).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
mkdir -p "${BUILD}"

# Required for FireNet path: export TENSORRT_ROOT=/path/to/TensorRT-10.x
# Example: export TENSORRT_ROOT="$HOME/TensorRT-10.8.0.43"
cmake -S "${ROOT}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  ${TENSORRT_ROOT:+-DTENSORRT_ROOT="${TENSORRT_ROOT}"}

cmake --build "${BUILD}" -j"$(nproc)"
echo "Binaries in ${BUILD}/"
echo "  ${BUILD}/phase0_smoke"
echo "  ${BUILD}/phase0_run"
echo "  ${BUILD}/build_engine   (if TensorRT found)"
echo ""
echo "Without TENSORRT_ROOT: smoke + proxy recon only."
echo "With TensorRT: export ONNX → build_engine → phase0_run --engine engines/firenet.engine"

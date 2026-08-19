#!/usr/bin/env bash
# Release build with Ninja.
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
echo "  ${BUILD}/eventgate_smoke"
echo "  ${BUILD}/eventgate"
echo "  ${BUILD}/build_engine   (if TensorRT found)"
echo ""
echo "Without TENSORRT_ROOT: smoke + voxel proxy only."
echo "With TensorRT: export ONNX, then build_engine, then eventgate --engine ..."

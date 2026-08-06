#!/usr/bin/env bash
# System deps for C++ Phase 0 (needs sudo). You run this.
set -eu

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  pkg-config \
  libhdf5-dev \
  libopencv-dev \
  wget \
  curl

echo "System deps OK. TensorRT 10.x: download from NVIDIA (tar or deb) and set TENSORRT_ROOT."
echo "  export TENSORRT_ROOT=/path/to/TensorRT-10.x"
echo "  export LD_LIBRARY_PATH=\$TENSORRT_ROOT/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Metavision: only needed on capture laptop for RAW→HDF5 conversion,"
echo "  OR on this machine if you convert .raw here. Prefer convert on capture host → rsync HDF5."

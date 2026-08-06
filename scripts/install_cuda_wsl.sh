#!/usr/bin/env bash
# Install CUDA 12.8 toolkit on WSL2 Ubuntu (needs sudo).
# You run this — AI will not.
set -eu

CUDA_VERSION="${CUDA_VERSION:-12-8}"
TMP=$(mktemp -d)
cd "$TMP"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y "cuda-toolkit-${CUDA_VERSION}"

CUDA_HOME="/usr/local/cuda"
if ! grep -q "libeventgate cuda" "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ""
    echo "# libeventgate cuda"
    echo "export PATH=${CUDA_HOME}/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
    echo "export CUDA_HOME=${CUDA_HOME}"
  } >> "$HOME/.bashrc"
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
nvcc --version
nvidia-smi
echo "CUDA toolkit OK — open a new shell or: source ~/.bashrc"

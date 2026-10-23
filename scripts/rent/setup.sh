#!/usr/bin/env bash
# scripts/rent/setup.sh: a fresh Ubuntu 22.04/24.04 box with an NVIDIA driver and CUDA 13.x (nvcc on PATH or in /usr/local/cuda/bin) and
# RTX 5080 / 5090 GPUs (sm_120a): build deps, the ufsm repo at BRANCH (default dino: the SSL batch), build, a one-minute GPU self-test
# (tests/test_dinor: the SSL kernels, toy DINO / InfoNCE loops). Run in the directory that will hold ufsm/ and runs/.
#   bash setup.sh            (REPO / BRANCH env to override; a private repo: REPO=https://<token>@github.com/SuperOptimizer/ufsm.git)
set -eu
REPO=${REPO:-https://github.com/SuperOptimizer/ufsm.git}; BRANCH=${BRANCH:-dino}
export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc > /dev/null || { echo "setup: nvcc not found (install CUDA 13.x)"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
sudo apt-get update -q && sudo apt-get install -y -q build-essential git pkg-config jq rsync libcurl4-openssl-dev libzstd-dev libblosc-dev zlib1g-dev libssl-dev libjpeg-dev
[ -d ufsm ] || git clone -q "$REPO" ufsm
cd ufsm && git fetch -q origin "$BRANCH" && git checkout -q "origin/$BRANCH" && git log --oneline -1
make -j"$(nproc)" build/ufsm build/test_dinor 2>&1 | tail -2
echo "setup: self-test (tests/test_dinor) ..."
timeout 600 ./build/test_dinor | tail -3
echo "setup: done. Next: bash ufsm/scripts/rent/run_arm.sh <name> (names: ufsm/scripts/rent/sweep.tsv)"

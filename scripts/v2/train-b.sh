#!/usr/bin/env bash
# v2 arm B (desk GPU 1): new architecture from scratch, 7 levels 16..256 (the ink96 shape: the larger receptive field for packed
# regions), same data and heads as arm A, P512 lean fp4, longer schedule. Usage: train-b.sh [NAME v2b] [STEPS 150000]
D=$(cd "$(dirname "$0")" && pwd)
export NAME=${1:-v2b} STEPS=${2:-150000} GPU=${GPU:-1}
export ARCH="--widths 16,16,32,64,128,192,256"
export INIT=""
export LRARGS="--warmup 2000 --muon-lr ${MLR:-0.02} --lr ${ALR:-0.001}"
exec bash $D/train-common.sh

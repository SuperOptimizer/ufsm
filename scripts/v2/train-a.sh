#!/usr/bin/env bash
# v2 arm A (desk GPU 0): warm start from wind96-wk1-best (task unwrap v1 donor: phase / normal / density rows -> unwrap2 0-8, winding
# mod 4 + the ink row fresh; ink row zero weights, bias -2), wind96 architecture (6 levels 32..192, dec 0,32, scale-cond), trunk LR x0.25
# for the first 5k steps (the heads at the full rates), P512 lean fp4. Usage: train-a.sh [NAME v2a] [STEPS 60000]
# Measured P/memory: see the memory note ufsm-v2-train-2026-10-31.md (laptop dry run).
D=$(cd "$(dirname "$0")" && pwd)
export NAME=${1:-v2a} STEPS=${2:-60000} GPU=${GPU:-0}
export ARCH="--widths 32,64,96,128,160,192 --dec-widths 0,32 --scale-cond 1"
export INIT="--init-from ${DONOR:-/vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt}"
export LRARGS="--warmup 500 --muon-lr ${MLR:-0.01} --lr ${ALR:-0.0005} --trunk-lr-mult ${TLR:-0.25} --trunk-lr-steps ${TLS:-5000}"
exec bash $D/train-common.sh

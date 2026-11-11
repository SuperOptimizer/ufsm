#!/usr/bin/env bash
# wzsmoke.sh K NAME STEPS: the latent winding run of the desk's runs/wzlo2 (wz896 recipe: P 896, widths 96..192, unwrap2 heads, latent
# grid 4 / KLT basis / sub-cell head, replay 3, wind96 warm start with --init-lr-mult 0.2) on a datakit pod, sources K/sources-wind-gt-pod.json.
# Binary: branch initlr-20261111 (U). Run dir K/runs/NAME.
set -u; K=$1; NAME=$2; STEPS=${3:-200}; U=${U:-/workspace/lat/build/ufsm}; L=/workspace/lat; mkdir -p $K/runs
T=$K/runs/train.sh; sed 's/--fp4 2 --gpu-aug 2 --input-prec 8 /--fp4 2 --gpu-aug 3 /' $L/scripts/rv/train.sh > $T
grep -q -- '--fp4 2 --gpu-aug 3 --mem' $T || { echo "train.sh changed: the flag patch did not apply"; exit 1; }
cd $K/runs && R=$K/runs U=$U SRCJ=$K/sources-wind-gt-pod.json LOCK= P=${P:-896} WORKERS=${WORKERS:-12} SLOTS=${SLOTS:-12} VALB=0 CKPT=500 SNAP=1000 \
  MAXFAIL=2 MINMEM=10 GPU=0 ARCH="--widths 96,128,160,192" WARM=300 MLR=0.01 ALR=0.0005 \
  HEADS="--task unwrap2 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1" \
  EXTRA="--lat 4 --lat-basis $K/small/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --replay 3 --init-from $K/small/wind96-wk1-best.ckpt --init-shift 2 --init-lr-mult 0.2 ${EXTRA:-}" \
  bash $T $NAME $STEPS

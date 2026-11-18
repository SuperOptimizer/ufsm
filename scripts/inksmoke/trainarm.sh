#!/usr/bin/env bash
# scripts/inksmoke/trainarm.sh ARM SOURCES.json: one arm of the latent-ink smoke (2026-11-18, the lead-approved retrain plan, user: smoke
# only): the u1 trunk (INIT, u1-13000-w) + its ink row, ink-only windows (human A / human + ink-teach B), the memorisation fixes:
# --replay 1, --fstem-drop 0 (predict always runs the stem), --ink-radial-drop 0.15 (both axis modes in distribution), --ink-mil-r-um 50
# (the MIL reach in um, not voxels), rungs 4.8 / 9.6 weighted 1:3 (production reads 9.6), --trunk-lr-mult 0.2 (no geometry targets in
# these windows); STEPS (3000), snapshots every SNAP (1000). Through scripts/univ/u1.train.sh (auto-resume, LOCK held for the run).
# Env: R (runs dir), U (binary), INIT, LATB (klt1.lcb), GPU (0), LOCK (the pod's /root/gpu.lock), P (1024), WORKERS / SLOTS (2 / 2: a 31 GB pod),
#      STEPS, SNAP, MINMEM (12), XTRA (more train flags), TARCH (the trunk's architecture flags, default u1's "--widths 96,128,160,192"), TXTRA
#      (more architecture / normalisation flags the INIT net needs), TUW (--uw, 1). From try9 (scratch2, 11-18 user: the ink retrain starts from
#      try9, not u1): TARCH="--widths 256,320,384,448,288,288" TXTRA="--attn-blocks 4 --attn-pe scroll --gn-ema 1 --gn-ema-warm 1000000000
#      --val-gn-live 1 --scan-norm 1 --gfilm 1 --fib-head 1 --muon-tc 2 --normuon 1" TUW=0 (its own flags), P a multiple of 128 (nlev 6).
set -u; ARM=${1:?ARM}; SRCJ=${2:?sources}; D=$(dirname "$(readlink -f "$0")")
: "${R:?R}" "${U:?U}" "${INIT:?INIT}" "${LATB:?LATB: the latent basis klt1.lcb (u1 was trained with it: without it the net has 4 inputs, not 32, and --init-from fails)}"
EXTRA="--lat 4 --lat-basis $LATB --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --replay 1 --um-rungs 4.8,9.6:3 --rung-shift-p 0.3 --film 2 \
 --fstem 16 --fstem-drop 0 --uw ${TUW:-1} --init-from $INIT --ink-head 1 --ink-mil 1 --ink-mil-r-um 50 --ink-dice 0 --ink-smooth 0.1 --ink-bias -1 \
 --ink-radial-drop 0.15 --ink-cache-mb ${INKCACHE:-1024} --trunk-lr-mult 0.2 --mix-every 250 ${TXTRA:-} ${XTRA:-}"
mkdir -p $R
R=$R U=$U SRCJ=$SRCJ GPU=${GPU:-0} LOCK=${LOCK-} P=${P:-1024} ARCH="${TARCH:---widths 96,128,160,192}" WORKERS=${WORKERS:-2} SLOTS=${SLOTS:-2} MEMMODE=auto \
  ALR=0.0001 MLR=0.002 WIND4L=0.25 ISO=64 MINMEM=${MINMEM:-12} SNAP=${SNAP:-1000} CKPT=${CKPT:-500} WARM=50 LOGE=10 RELAUNCH=${RELAUNCH:-3} \
  HEADS="--task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1" \
  EXTRA="$EXTRA" bash $D/../univ/u1.train.sh inksmoke$ARM ${STEPS:-3000}

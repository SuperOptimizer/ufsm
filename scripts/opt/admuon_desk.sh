#!/bin/bash
# Adaptive Muon A/B on the desk, scroll data (DESIGN.md 2026-11-14 "Fully adaptive Muon"): one arm of the scratch2 net on one 5060 Ti at
# P 512, try6's sources / heads / augmentation / XFLAGS, --init-from one fixed try6 checkpoint, --det 1 (the same windows in every arm
# of a seed), --accum 5, Muon LR (env, 0.003) / AdamW LR / 10, WSD warmup 50, cooldown 0.2.
# Usage: admuon_desk.sh GPU ARM STEPS SEED   (env: U binary, INIT start checkpoint, OUT dir, WORKERS 8, SLOTS 10)
# ARM: muon | muon2x | normuon | lars | nmlars | dog | adaptive. Holds /tmp/ufsm-suite-locks/gpu<GPU>.excl for its lifetime (tests stay off the card).
# Start detached: setsid nohup scripts/opt/admuon_desk.sh 0 muon 2000 7 > /dev/null 2>&1 &
set -u
GPU=$1 ARM=$2 STEPS=$3 SEED=$4
U=${U:?binary} INIT=${INIT:-/vesuvius/ufsm/admuon/start.ckpt} OUT=${OUT:-/vesuvius/ufsm/admuon/ab}
LR=${LR:-0.003}   # the base Muon lr (try6 since 11:59: 0.003 / AdamW 3e-4); AdamW at LR / 10
MLR=$LR
case "$ARM" in
  muon) AF=() ;;
  muon2x) AF=(); MLR=$(awk "BEGIN{print 2 * $LR}") ;;
  normuon) AF=(--normuon 1) ;;
  lars) AF=(--muon-lars 1) ;;
  nmlars) AF=(--normuon 1 --muon-lars 1) ;;   # (a) + (b) without the DoG step
  dog) AF=(--muon-dog 1) ;;
  adaptive) AF=(--muon-adaptive 1) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
mkdir -p "$OUT"; R="$OUT/r_${ARM}_lr${LR}_s$SEED"
case "$R" in /vesuvius/ufsm/admuon/*/r_*) rm -rf -- "$R" ;; *) echo "refusing OUT $R" >&2; exit 2 ;; esac
AUG="--zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-noise-p 0.15 --aug-cnoise-p 0.1
  --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1"
exec 9> /tmp/ufsm-suite-locks/gpu$GPU.excl; flock 9
cd /vesuvius/ufsm/v2/lat/runs || exit 1
export MALLOC_ARENA_MAX=4 UFSM_HEAD_SLAB_Z=16 UFSM_MIL_ASYNC=0 CUDA_VISIBLE_DEVICES=$GPU
/home/forrest/bin/memcap -t "$U" train /vesuvius/ufsm/univ/scratch1/sources-scratch1-ph80.json --out "$R" --gpus 0 --P 512 --B 1 \
  --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1 --fibre-gpu 0 --ink-gpu 0 \
  --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1 \
  --steps "$STEPS" --sched wsd --cooldown 0.2 --opt muon --fp4 2 --gpu-aug 3 --mem auto --val-store disk --workers ${WORKERS:-8} \
  --sampler-slots ${SLOTS:-10} --log-every 10 --opt-stats 50 --val-every 1000000 --val-batches 0 --ckpt-every 1000000 --seed "$SEED" --val-seed 1 \
  --warmup 50 --muon-lr $MLR --lr $(awk "BEGIN{print $LR / 10}") $AUG \
  --lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --um-rungs 4.8,9.6:2,19.2 \
  --rung-shift-p 0.3 --film 2 --fstem 16 --fstem-drop 0.25 --uw 0 --ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1 \
  --ink-cache-mb 2048 --onfly-slots 2 --onfly-mb 768 --strict-sources 1 --muon-tc 2 \
  --gn-ema-warm 1000000000 --skip-empty 1 --skip-thr 0 --skip-margin 1 --accum 5 \
  --det 1 --init-from "$INIT" "${AF[@]}" > "$R.log" 2>&1
echo "arm $ARM seed $SEED rc $?" >> "$R.log"

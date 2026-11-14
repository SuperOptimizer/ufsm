#!/bin/bash
# Adaptive Muon A/B (DESIGN.md 2026-11-14 "Fully adaptive Muon"): one arm of the scratch2 net single card at P 512 on the wind fixture,
# --det 1 / one seed (the same window stream for every arm), Muon 0.006 / AdamW 6e-4 (try6's), WSD with a 0.2 cooldown, optionally
# --init-from a checkpoint, try6's XFLAGS (--gn-ema-warm 1e9: batch GroupNorm statistics throughout; the default 500 switched to running stats at
# step 501 and set off a blow-up in the first control: gn 9 -> 98, loss 1.27 -> 5.6 by step 630), --accum 5 (ACCUM; try6: 10 windows / update). Usage: admuon_ab.sh BIN OUTDIR ARM STEPS SEED [INIT_CKPT] [extra flags...]
# Env: P (512), LATB (klt1.lcb), OST (--opt-stats, 50). Run from the directory holding fx/ (the wind fixture).
# ARM: muon | normuon | lars | adaptive (a+b+c) | dog. Run under the pod's GPU lock:
#   flock /root/gpu.lock timeout -k 10 3600 scripts/opt/admuon_ab.sh build/ufsm run muon 2000 7 try6.ckpt
set -u
BIN=$1 OUT=$2 ARM=$3 STEPS=$4 SEED=$5 INIT=${6:-}; shift 5; [ $# -gt 0 ] && shift
case "$ARM" in
  muon) AF=() ;;
  normuon) AF=(--normuon 1) ;;
  lars) AF=(--muon-lars 1) ;;
  dog) AF=(--muon-dog 1) ;;
  adaptive) AF=(--muon-adaptive 1) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
INITF=(); [ -n "$INIT" ] && INITF=(--init-from "$INIT")
mkdir -p "$OUT"; R="$OUT/r_${ARM}_s$SEED"
case "$R" in "$OUT"/r_*) rm -rf -- "$R" ;; esac
env UFSM_HEAD_SLAB_Z=16 CUDA_VISIBLE_DEVICES=0 "$BIN" train fx/sources-ink.json --out "$R" --gpus 0 --P ${P:-512} --B 1 \
  --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1 --fibre-gpu 1 --ink-gpu 1 \
  --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --normal-step 1 \
  --lat 4 --lat-basis ${LATB:-klt1.lcb} --lat-c 29 --lat-step 0.25 --lat-guide 3 --film 2 --fstem 16 --fstem-drop 0.25 --uw 0 \
  --ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1 --zfix 1 --aug none \
  --steps "$STEPS" --sched wsd --cooldown 0.2 --warmup 50 --opt muon --muon-lr 0.006 --lr 0.0006 --muon-tc 2 \
  --fp4 2 --gpu-aug 3 --mem auto --workers 6 --sampler-slots 4 --log-every 10 --opt-stats ${OST:-50} --val-batches 0 --val-every 1000000 \
  --gn-ema-warm 1000000000 --skip-empty 1 --skip-thr 0 --skip-margin 1 --accum ${ACCUM:-5} \
  --ckpt-every 1000000 --seed "$SEED" --det 1 "${INITF[@]}" "${AF[@]}" "$@" > "$R.log" 2>&1
echo "arm $ARM seed $SEED rc $?" >> "$R.log"

#!/bin/bash
# mab_arm.sh ARM SEED (2026-11-15, Muon improvements batch 1, DESIGN.md "Muon update scaling" / "Cautious weight decay"): one from-scratch
# arm of the scratch2 net on card 1 at P ${P:-256}, ${STEPS:-1500} windows, --det 1, try6/try7's sources / heads / aug / XFLAGS, NorMuon,
# WSD warmup ${WARM:-150} cooldown 0.2, --accum 5, a held-out validation at the end. Shares card 1 with the suite: gpu1.excl SHARED plus
# one slot lock. The run directory (its 2 GB final checkpoint) is deleted afterwards; the log stays.
# ARM: base | rms | wd10 | cwd10   (env LR: Muon lr, default 0.003, AdamW LR / 10; MLR overrides the Muon lr only)
set -u
ARM=$1 SEED=$2
U=${U:?binary} OUT=/vesuvius/ufsm/admuon/mab LR=${LR:-0.003} MLR=${MLR:-${LR:-0.003}} STEPS=${STEPS:-1500} P=${P:-256}
case "$ARM" in
  base) AF=() ;;
  rms) AF=(--muon-scale rms --muon-rms 0.2) ;;
  wd10) AF=(--muon-wd-mult 10) ;;
  cwd10) AF=(--muon-wd-mult 10 --muon-cwd 1) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
R="$OUT/${ARM}_m${MLR}_p${P}_s$SEED"
case "$R" in /vesuvius/ufsm/admuon/mab/?*) rm -rf -- "$R" ;; *) echo "refusing OUT $R" >&2; exit 2 ;; esac
AUG="--zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-noise-p 0.15 --aug-cnoise-p 0.1
  --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1"
L=/tmp/ufsm-suite-locks
exec 8> $L/gpu1.excl; flock -s 8
while :; do got=; for s in 5 4 3 2 1 0; do exec 9> $L/gpu1.$s; if flock -n 9; then got=$s; break; fi; exec 9>&-; done; [ -n "$got" ] && break; sleep 10; done
cd /vesuvius/ufsm/v2/lat/runs || exit 1
export MALLOC_ARENA_MAX=4 UFSM_HEAD_SLAB_Z=16 UFSM_MIL_ASYNC=0 CUDA_VISIBLE_DEVICES=1
/home/forrest/bin/memcap -t "$U" train /vesuvius/ufsm/univ/scratch1/sources-scratch1-ph80.json --out "$R" --gpus 0 --P "$P" --B 1 \
  --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1 --fibre-gpu 0 --ink-gpu 0 \
  --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1 \
  --steps "$STEPS" --sched wsd --cooldown 0.2 --opt muon --normuon 1 --fp4 2 --gpu-aug 3 --mem auto --val-store disk --workers ${WORKERS:-8} \
  --sampler-slots ${SLOTS:-10} --log-every 10 --opt-stats 50 --val-every "$STEPS" --val-batches ${VALB:-24} --ckpt-every 1000000 --seed "$SEED" --val-seed 1 \
  --warmup ${WARM:-150} --muon-lr $MLR --lr $(awk "BEGIN{print $LR / 10}") $AUG \
  --lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --um-rungs 4.8,9.6:2,19.2 \
  --rung-shift-p 0.3 --film 2 --fstem 16 --fstem-drop 0.25 --uw 0 --ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1 \
  --ink-cache-mb 2048 --onfly-slots 2 --onfly-mb 768 --strict-sources 1 --muon-tc 2 \
  --gn-ema-warm 1000000000 --skip-empty 1 --skip-thr 0 --skip-margin 1 --accum 5 \
  --det 1 "${AF[@]}" > "$R.log" 2>&1
rc=$?
echo "arm $ARM seed $SEED mlr $MLR rc $rc" >> "$R.log"
case "$R" in /vesuvius/ufsm/admuon/mab/?*) rm -rf -- "$R" ;; esac

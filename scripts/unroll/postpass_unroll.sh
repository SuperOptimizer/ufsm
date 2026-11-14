#!/usr/bin/env bash
# postpass_unroll.sh [GPU]: the unroll audit's GPU measurements, queued for after the mega test pass (MEGA.md: no GPU before it). One
# card, in a desk GPU window (scratch2 stopped) or on a pod; each item is resumable and writes only its own output dirs. ~2 h.
#   #9  scripts/unroll/fith2_prof.sh: the fused --fit-h 2 kernel (UFSM_PRED_FIT_FUSED_SMALLH) identity + speed -> decide its default
#   #5  scripts/unroll/seam_ab.sh: tile-border seams of the fit fields, 640 vs 1152 windows vs two grids + field-blend -> FBLEND default
#   #4  scripts/inkgt/u1667ink.sh LATINK_CK pp-lv2 (LV=2): held-out 1667 ink AUC with --axis (the radial channels fixed, unroll #3)
#   #21 the same at LV=1 (pp-lv1, 4.8 um): render resolution, AUC lv1 vs lv2 on the same labels
# (#8 the slab-vs-ref coherence recheck belongs to pub1667 after fibfit's dir -1 refit.) Results: the items' own logs / tsvs, a line each
# in $OUT/postpass.log. The ink items run on GPU 0 whatever GPU says (u1667ink.sh: flock /tmp/gpu0-bench.lock) and write under
# /vesuvius/ufsm/unroll/PHerc1667-u1/ink{,r}/pp-lv* + sched/inkscore-u1-pp-lv*.tsv. Env: OUT, ITEMS ("9 5 4 21"), LATINK_CK (inkpod 15000), U.
set -u
G=${1:-0}; REPO=$(cd "$(dirname "$0")/../.." && pwd); OUT=${OUT:-/vesuvius/ufsm/tmp-postpass-unroll}; mkdir -p "$OUT"; L=$OUT/postpass.log
CKI=${LATINK_CK:-/vesuvius/ufsm/v2/lat/runs/inkpod/snap/15000.ckpt}; export U=${U:-$REPO/build/ufsm}
log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
for it in ${ITEMS:-9 5 4 21}; do
  t0=$(date +%s)
  case $it in
    9) OUT=$OUT/fith2 bash "$REPO/scripts/unroll/fith2_prof.sh" $G > "$OUT/fith2.log" 2>&1; rc=$?; r=$(grep -E 'IDENTICAL|DIFFER|bench h' "$OUT/fith2/summary.txt" 2>/dev/null | tr '\n' ' ') ;;
    5) W=$OUT/seamab GPU=$G bash "$REPO/scripts/unroll/seam_ab.sh" > "$OUT/seamab.log" 2>&1; rc=$?; r=$(tail -3 "$OUT/seamab.log" | tr '\n' ' ') ;;
    4) CUDA_VISIBLE_DEVICES=$G LV=2 bash "$REPO/scripts/inkgt/u1667ink.sh" "$CKI" pp-lv2 > "$OUT/ink-lv2.log" 2>&1; rc=$?; r=$(tail -2 "$OUT/ink-lv2.log" | tr '\n' ' ') ;;
    21) CUDA_VISIBLE_DEVICES=$G LV=1 bash "$REPO/scripts/inkgt/u1667ink.sh" "$CKI" pp-lv1 > "$OUT/ink-lv1.log" 2>&1; rc=$?; r=$(tail -2 "$OUT/ink-lv1.log" | tr '\n' ' ') ;;
    *) log "unknown item $it"; continue ;;
  esac
  log "unroll#$it rc $rc, $(( $(date +%s) - t0 )) s: $r"
done

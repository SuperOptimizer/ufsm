#!/usr/bin/env bash
# predict --tta on a latent-grid net (half_res 3, the sub-cell head --lat-guide 3, fit fields through --fit-zarr; src/predict.c tta_pass lat):
#  1. --tta 2 with UFSM_TTA_ORDER=0 (the identity again, through the augmented-pass path: CT window from the saved copy, the cell input,
#     the trunk, the heads into the whole-window planes) == --tta 1 on the same whole-window path (UFSM_PRED_FIT_SLAB=0), every field
#     file byte for byte: each pass pools the same accumulators, so a repeated pass changes nothing.
#  2. --tta 8 runs (it was refused before: "needs the CT-stem device path"), prints its startup line, and its fields differ from --tta 1.
# Both for the plain net (the direct cell input) and a fine-stem net (train --fstem 16, as u1 / try9: the pass takes the plain pass's xd
# route, nn_pred_input_d + the pooled input, and adds the stem from xd; the pod u1 / pab-A --tta 8 arms were refused 11-18 without it).
# Usage: test_tta_lat.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
case "$D" in /*/*) ;; *) echo "FAIL: DIR must be an absolute path below /"; exit 1;; esac
rm -rf -- "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
for S in s f; do FS=; [ $S = f ] && FS="--fstem 16"
  "$U" train fx/sources.json --out ${S} --task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 \
       --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3 \
       --lat 4 --lat-guide 3 $FS > ${S}.s.log 2>&1 || fail "train ($D/${S}.s.log)"
  P=(--axis fx/axis.json --um 9.6 --level 0 --window 128 --halo 32 --shard 128 --levels 1 --threads 2 --box 0,32,32,192,192,192 --q 0 --no-out 1)
  UFSM_PRED_FIT_SLAB=0 "$U" predict ${S}/last.ckpt fx ct ${S}p1 --fit-zarr ${S}p1/fz "${P[@]}" > ${S}p1.log 2>&1 || fail "predict --tta 1 ($D/${S}p1.log)"
  UFSM_TTA_ORDER=0 "$U" predict ${S}/last.ckpt fx ct ${S}p2 --fit-zarr ${S}p2/fz --tta 2 "${P[@]}" > ${S}p2.log 2>&1 || fail "predict --tta 2 order 0 ($D/${S}p2.log)"
  grep -q -- '--tta 2 on the latent-grid net' ${S}p2.log || fail "no latent --tta startup line ($D/${S}p2.log)"
  n=$(find ${S}p1/fz -type f | wc -l); [ "$n" -gt 5 ] || fail "only $n field files"
  diff -r ${S}p1/fz ${S}p2/fz > ${S}.d2.txt || fail "--tta 2 (identity twice) differs from --tta 1 ($(wc -l < ${S}.d2.txt) lines, $D/${S}.d2.txt)"
  "$U" predict ${S}/last.ckpt fx ct ${S}p8 --fit-zarr ${S}p8/fz --tta 8 "${P[@]}" > ${S}p8.log 2>&1 || fail "predict --tta 8 ($D/${S}p8.log)"
  diff -rq ${S}p1/fz ${S}p8/fz > /dev/null && fail "--tta 8 fields identical to --tta 1 (the passes did not reach the accumulators)"
done
echo "test_tta_lat: OK (plain + fine-stem net, $n field files: identity repeated = --tta 1 byte for byte; --tta 8 runs and differs)"
cd /; rm -rf -- "$D"

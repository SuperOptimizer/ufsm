#!/usr/bin/env bash
# predict --tta on a latent-grid net (half_res 3, the sub-cell head --lat-guide 3, fit fields through --fit-zarr; src/predict.c tta_pass lat):
#  1. --tta 2 with UFSM_TTA_ORDER=0 (the identity again, through the augmented-pass path: CT window from the saved copy, the cell input,
#     the trunk, the heads into the whole-window planes) == --tta 1 on the same whole-window path (UFSM_PRED_FIT_SLAB=0), every field
#     file byte for byte: each pass pools the same accumulators, so a repeated pass changes nothing.
#  2. --tta 8 runs (it was refused before: "needs the CT-stem device path"), prints its startup line, and its fields differ from --tta 1.
# Usage: test_tta_lat.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
case "$D" in /*/*) ;; *) echo "FAIL: DIR must be an absolute path below /"; exit 1;; esac
rm -rf -- "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
"$U" train fx/sources.json --out s --task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 \
     --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3 \
     --lat 4 --lat-guide 3 > s.log 2>&1 || fail "train ($D/s.log)"
P=(--axis fx/axis.json --um 9.6 --level 0 --window 128 --halo 32 --shard 128 --levels 1 --threads 2 --box 0,32,32,192,192,192 --q 0 --no-out 1)
UFSM_PRED_FIT_SLAB=0 "$U" predict s/last.ckpt fx ct p1 --fit-zarr p1/fz "${P[@]}" > p1.log 2>&1 || fail "predict --tta 1 ($D/p1.log)"
UFSM_TTA_ORDER=0 "$U" predict s/last.ckpt fx ct p2 --fit-zarr p2/fz --tta 2 "${P[@]}" > p2.log 2>&1 || fail "predict --tta 2 order 0 ($D/p2.log)"
grep -q -- '--tta 2 on the latent-grid net' p2.log || fail "no latent --tta startup line ($D/p2.log)"
n=$(find p1/fz -type f | wc -l); [ "$n" -gt 5 ] || fail "only $n field files"
diff -r p1/fz p2/fz > d2.txt || fail "--tta 2 (identity twice) differs from --tta 1 ($(wc -l < d2.txt) lines, $D/d2.txt)"
"$U" predict s/last.ckpt fx ct p8 --fit-zarr p8/fz --tta 8 "${P[@]}" > p8.log 2>&1 || fail "predict --tta 8 ($D/p8.log)"
diff -rq p1/fz p8/fz > /dev/null && fail "--tta 8 fields identical to --tta 1 (the passes did not reach the accumulators)"
echo "test_tta_lat: OK ($n field files: identity repeated = --tta 1 byte for byte; --tta 8 runs and differs)"
cd /; rm -rf -- "$D"

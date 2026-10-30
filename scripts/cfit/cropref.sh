#!/usr/bin/env bash
# cropref.sh FIELDS A|B OUT [GPU]: the VF Paris 4 crop benchmark (2026-10-29) for any whole-scroll Paris 4 fit-field set on the
# wind96 field grid (h 2, 9.6 um; e.g. desk /vesuvius/ufsm/paris4/fields/wind96, forlindesk /vesuvius/ufsm/vf/p4/fields/wind96).
#   crop A = field rows 4544..5952, crop B = 6016..7424 (1408 rows each), y 608 + 3328, x 352 + 3520 (the cfit union crop)
#   ref    = count-field --R 0 --pool 8 --phase-w 4 --iters 1200 --model-only 1 --model-stride STRIDE (1) [the cf coarse ref]
#   score  = fit --model ref --codes Q 9.6 --score-pairs 1 --score-detail 1 (q-9p6 Paris 4 GT)
# wind96 numbers (ids | pairs 0.5 / 4 mm): A stride 1 .1766 | .651 / .238 (stride 4 .1770 | .666 / .242); B stride 1 .1537 | .7285 / .3087.
# Env: U (ufsm binary), Q (q-9p6.zarr root; desk default), TH (12), MEMGB (24). Prints one summary line; OUT/ref, OUT/ref.log, OUT/score.log.
set -u; F=$1; W=$2; O=$3; G=${4:-0}
U=${U:-ufsm}; Q=${Q:-/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr}
case $W in A) Z0=4544;; B) Z0=6016;; *) echo "crop A or B"; exit 2;; esac
mkdir -p $O; [ -e $O/f/fields.json ] || bash $(dirname "$0")/../wbench/mkcrop.sh $F $O/f $Z0 608 352 1408 3328 3520 > /dev/null
[ -e $O/ref/model.json ] || nice -n 5 $U count-field $O/f $O/unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $O/ref --model-only 1 \
  --model-stride ${STRIDE:-1} --threads ${TH:-12} --mem-gb ${MEMGB:-24} > $O/ref.log 2>&1 || { echo "count-field failed: $O/ref.log"; exit 1; }
$U fit $O/f --model $O/ref --gpu $G --codes $Q 9.6 --score-pairs 1 --score-detail 1 > $O/score.log 2>&1 || { echo "score failed: $O/score.log"; exit 1; }
echo "$O crop $W: $(grep -oE 'winding ids right [0-9.]+' $O/score.log | tail -1) | pairs $(grep -oE '0.5 mm [0-9.]+|4 mm [0-9.]+' $O/score.log | head -2 | tr '\n' ' ')"

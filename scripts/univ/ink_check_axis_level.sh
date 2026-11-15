#!/bin/bash
# GPU check for the inkmil audit (#2, #3): one snapshot on the 34 ink crops at level 1 / 2, with and without the scroll's axis, then the per-scroll gate.
#   scripts/univ/ink_check_axis_level.sh CKPT TAG        (run in a GPU window; each cell is one eval.sh TASKS=ink pass: ~1 h on one card)
# Cells: L1+axis, L1 no axis (= the old numbers), L2+axis, L2 no axis. Outputs $OUT/<TAG>-L<lv>-<ax|noax>/ and the gate table $OUT/<TAG>.gate.txt.
# Reads: if L1 noax reproduces inkpod-10000's 0.795 / 0.729 / 0.715 / 0.634 / 0.539 the harness is right; (L1 ax - L1 noax) is finding #2's size,
# (L1 - L2) finding #3's. env as eval.sh (U, UP, CTROOT, INKC, INKGT, GPU ...).
set -u
CK=${1:?ckpt}; TAG=${2:?tag}; OUT=${OUT:-/vesuvius/ufsm/wbench/univ/inkcheck}; HERE=$(dirname $(readlink -f $0)); mkdir -p $OUT
for lv in 1 2; do for ax in 1 0; do
  E=$OUT/$TAG-L$lv-$([ $ax = 1 ] && echo ax || echo noax); mkdir -p $E
  env TASKS=ink INKP=heads LV_INK=$lv INK_AXIS=$ax INKC="${INKC:-/vesuvius/ufsm/inkgt/evcrops /vesuvius/ufsm/inkgt/evcrops-ho}" INKGT="${INKGT:-/vesuvius/ufsm/inkgt/evgt /vesuvius/ufsm/inkgt/evgt-ho}" \
    E=$E bash $HERE/eval.sh $CK $TAG > $E/eval.out 2>&1 || echo "cell $E failed (see $E/eval.out)"
done; done
{ for c in L1-ax L1-noax L2-ax L2-noax; do echo "== $c"; f=$(ls $OUT/$TAG-$c/*/ink/vscores.tsv 2>/dev/null | head -1); [ -n "$f" ] && python3 -I $HERE/ink_gate.py $f $(ls $OUT/$TAG-L1-noax/*/ink/vscores.tsv 2>/dev/null | head -1); done; } > $OUT/$TAG.gate.txt 2>&1
cat $OUT/$TAG.gate.txt

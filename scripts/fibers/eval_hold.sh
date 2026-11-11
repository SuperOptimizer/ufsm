#!/usr/bin/env bash
# scripts/fibers/eval_hold.sh RUN [CKPT]: a validation of the H / V fibre mask model that does not use its own labels: on the Paris 4
# HOLD box (wbench holdout, another scanner setting: 2.4 um scan at level 2 = 9.6 um; never in training) predict both channels, trace
# lines through each mask (ufsm fiber-trace --mask: the axis from the mask's structure tensor) and ask the fibre judge how often a
# traced line stays in one sheet of the human GT winding (ufsm fiber-judge --phase). A mask that marks fibres gives long lines that
# stay in their sheet; one that marks papyrus gives short lines or lines that wander. Reference rows on the same box (2026-11-10,
# REF=1 recomputes them, CPU): the published fibres and the official fibre prediction traced the same way.
# Needs a GPU for the two predict passes (GPU, default 0); everything after is CPU. Output: RUN/eval/hold-<step>/{H,V} + judge.txt.
set -u
RUN=${1:?RUN dir}; CK=${2:-$RUN/last.ckpt}; U=${U:-/home/forrest/ufsm-fib/build/ufsm}; GPU=${GPU:-0}
CT=${CT:-/vesuvius/ufsm/v2/ct}; KEY=${KEY:-PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr}
# (pod: CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp KEY=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
#  PREDICT_ONLY=1: the two predicts only; copy RUN/eval/hold-STEP/{H,V} to the desk and rerun there for the trace + GT judge)
GT=/vesuvius/ufsm/gt/paris4-wind-20261019; W=/vesuvius/ufsm/fibers/work
BOX=12288,4096,3584,512,576,768          # HOLD in level-2 voxels (l0 49152,16384,14336,2048,2304,3072)
J="--um 2.4 --level 2 --slab 12288,512 --region 4096,4608,3584,4352 --tile 256 --d-mm 0.3,0.6,1.2,2.4"
if [ "${REF:-0}" = 1 ]; then
  S=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/fibers/20260411134726-fibers-20260915212757-L1
  K=PHercParis4-20260411134726-las-sd1-7ff0ce6c; C=/vesuvius/ufsm/fibers/raw/predcache
  nice -n 19 $U fiber-trace $S $K $W/hold_off.fib --box 3056,1008,880,160,176,224 --scaledown 4 --cache $C --threads 2 > $W/hold_off.log 2>&1
  nice -n 19 $U fiber-trace $S $K $W/hold_offpo.fib --box 3056,1008,880,160,176,224 --scaledown 4 --cache $C --threads 2 --presence-only 1 > $W/hold_offpo.log 2>&1
  echo "== published Paris4_1mm_v2 fibres";            nice -n 19 $U fiber-judge /vesuvius/ufsm/fibers/raw/Paris4_1mm_v2.afv --tr /vesuvius/ufsm/fibers/meta/p4_2026_to_2023.json --phase $GT phaseA/9.6 $J | tail -3
  echo "== official prediction, direction channels";   nice -n 19 $U fiber-judge $W/hold_off.fib --phase $GT phaseA/9.6 $J | tail -3
  echo "== official prediction, presence only";        nice -n 19 $U fiber-judge $W/hold_offpo.fib --phase $GT phaseA/9.6 $J | tail -3
  exit 0
fi
[ -f "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
st=$(head -c 200000 $CK | head -1 | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2); o=$RUN/eval/hold-${st:-x}; mkdir -p $o
for c in 0 1; do f=$([ $c = 0 ] && echo H || echo V)
  [ -e $o/$f/9.6/zarr.json ] || $U predict $CK $CT $KEY $o/$f --um 2.4 --level 2 --box $BOX --channel $c --gpu $GPU --threads 6 ${CACHE:+--cache $CACHE} > $o/predict_$f.log 2>&1 || { echo "predict $f failed: $o/predict_$f.log"; exit 1; }
  [ "${PREDICT_ONLY:-0}" = 1 ] && { echo "$f predicted: $o/$f"; continue; }
  nice -n 19 $U fiber-trace - - $o/$f.fib --mask $o/$f 9.6 --mask-origin ${BOX%,*,*,*} --box $BOX --scaledown 2 --family $f --half 1024 --min-len 48 --threads 4 > $o/trace_$f.log 2>&1 || { echo "trace $f failed"; exit 1; }
  tail -1 $o/trace_$f.log
  nice -n 19 $U fiber-judge $o/$f.fib --phase $GT phaseA/9.6 $J --out $o/judge_$f.tsv | tail -3
done | tee $o/judge.txt

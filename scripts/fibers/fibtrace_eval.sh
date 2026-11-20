#!/usr/bin/env bash
# scripts/fibers/fibtrace_eval.sh NAME HMASK VMASK [fiber-trace options...]: trace one fibre source on Paris 4 HOLD and score it (fibtrace-20261119).
#   HMASK / VMASK: mask stores (root with key 9.6, box-local: predict --box), e.g. fibhv-d's H / V or try9's fib head c14 / c15.
#   Per family: fiber-trace --mask (tube axis unless the options say otherwise), then
#     fiber-match against the open-data spiral dataset's fibres (native 2.4 um; HOLD lies outside Paris4_1mm_v2's z band): recall,
#       precision, runs (how far one of our lines follows one reference fibre), our lens (arc-median, >= 1 / 2 mm);
#     fiber-judge --phase against GT phaseA: persistent sheet changes per m, pairs within half a turn at 0.3 / 0.6 / 1.2 / 2.4 mm.
# Out: $O/NAME/{H,V}.fib, trace_*.log, match_*.out, judge_*.out and NAME.sum (one line per family). CPU only.
set -u
N=$1 HM=$2 VM=$3; shift 3
U=${U:-/home/forrest/ufsm-fibtrace/build/ufsm}; O=${O:-/vesuvius/ufsm/fibtrace/hold}; T=${T:-8}; mkdir -p $O/$N
BOX=12288,4096,3584,512,576,768; ORG=12288,4096,3584; REF=${REF:-/vesuvius/ufsm/fibers/raw/spiral_fibers.afv}
PHG=${PHG:-/vesuvius/ufsm/gt/paris4-phaseA-evalboxes-20261114}; PHK=${PHK:-phaseA/9.6}
J="--um 2.4 --level 2 --slab 12288,512 --region 4096,4608,3584,4352 --tile 256 --d-mm 0.3,0.6,1.2,2.4"
: > $O/$N.sum
for f in H V; do m=$HM; [ $f = V ] && m=$VM
  ~/bin/lowprio $U fiber-trace - - $O/$N/$f.fib --mask $m 9.6 --mask-origin $ORG --box $BOX --scaledown 2 --family $f --half 1024 --min-len 48 --threads $T "$@" > $O/$N/trace_$f.log 2>&1 || { echo "$N $f: trace failed" >> $O/$N.sum; continue; }
  ~/bin/lowprio $U fiber-match $O/$N/$f.fib $REF --box 49152,16384,14336,2048,2304,3072 --um 2.4 > $O/$N/match_$f.out 2>&1
  ~/bin/lowprio $U fiber-judge $O/$N/$f.fib --phase $PHG $PHK $J > $O/$N/judge_$f.out 2>&1
  lens=$(/usr/bin/grep "^  $f ours" $O/$N/match_$f.out | sed 's/ | ref.*//; s/^  . ours: //')
  mt=$(/usr/bin/grep "^  $f recall" $O/$N/match_$f.out | sed 's/^  . //; s/ ([0-9 /]* ref samples)//; s/ ([0-9 /]*)//')
  jd=$(/usr/bin/grep "^  $f:" $O/$N/judge_$f.out | sed -E 's/^  .: //; s/excursions [0-9]+ = [0-9.]+ \/m, //; s/ there-and-back [0-9.]+ \/m;//')
  echo "$N $f | $lens | $mt | $jd | $(/usr/bin/grep 'line ends' $O/$N/trace_$f.log | sed 's/.*kept lines): //')" >> $O/$N.sum
done
cat $O/$N.sum

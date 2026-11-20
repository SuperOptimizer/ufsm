#!/usr/bin/env bash
# scripts/fibers/fibtrace_eval.sh NAME HMASK VMASK [fiber-trace options...]: trace one fibre source on Paris 4 HOLD and score it (fibtrace-20261119).
#   HMASK / VMASK: mask stores (root with key 9.6, box-local: predict --box), e.g. fibhv-d's H / V or try9's fib head c14 / c15.
#   Per family: fiber-trace --mask (tube axis unless the options say otherwise), then
#     fiber-match against the open-data spiral dataset's fibres (native 2.4 um; HOLD lies outside Paris4_1mm_v2's z band): recall,
#       precision, runs (how far one of our lines follows one reference fibre), our lens (arc-median, >= 1 / 2 mm);
#     fiber-judge --phase against GT phaseA: persistent sheet changes per m, pairs within half a turn at 0.3 / 0.6 / 1.2 / 2.4 mm.
# Out: $O/NAME/{H,V}.fib, trace_*.log, match_*.out, judge_*.out and NAME.sum (one line per family). CPU only.
# Any failed step (missing mask, trace / match / judge rc != 0, an empty .fib, empty summary fields) writes a FAILED line and exits 1
# (2026-11-20, fullvram: failures used to leave a .sum with empty fields and exit 0).
set -u
N=$1 HM=$2 VM=$3; shift 3
U=${U:-/home/forrest/ufsm-fibtrace/build/ufsm}; O=${O:-/vesuvius/ufsm/fibtrace/hold}; T=${T:-8}; mkdir -p $O/$N
BOX=12288,4096,3584,512,576,768; ORG=12288,4096,3584; REF=${REF:-/vesuvius/ufsm/fibers/raw/spiral_fibers.afv}
PHG=${PHG:-/vesuvius/ufsm/gt/paris4-phaseA-evalboxes-20261114}; PHK=${PHK:-phaseA/9.6}
J="--um 2.4 --level 2 --slab 12288,512 --region 4096,4608,3584,4352 --tile 256 --d-mm 0.3,0.6,1.2,2.4"
: > $O/$N.sum.tmp; rc=0
fail() { echo "$N $f: FAILED: $*" | tee -a $O/$N.sum.tmp >&2; rc=1; }
for f in H V; do m=$HM; [ $f = V ] && m=$VM
  [ -e "$m/9.6/zarr.json" ] || { fail "no mask $m/9.6"; continue; }
  ~/bin/lowprio $U fiber-trace - - $O/$N/$f.fib.tmp --mask $m 9.6 --mask-origin $ORG --box $BOX --scaledown 2 --family $f --half 1024 --min-len 48 --threads $T "$@" > $O/$N/trace_$f.log 2>&1 \
    && [ -s $O/$N/$f.fib.tmp ] && mv $O/$N/$f.fib.tmp $O/$N/$f.fib || { fail "trace ($O/$N/trace_$f.log)"; continue; }
  ~/bin/lowprio $U fiber-match $O/$N/$f.fib $REF --box 49152,16384,14336,2048,2304,3072 --um 2.4 > $O/$N/match_$f.out 2>&1 || { fail "fiber-match ($O/$N/match_$f.out)"; continue; }
  ~/bin/lowprio $U fiber-judge $O/$N/$f.fib --phase $PHG $PHK $J > $O/$N/judge_$f.out 2>&1 || { fail "fiber-judge ($O/$N/judge_$f.out)"; continue; }
  lens=$(/usr/bin/grep "^  $f ours" $O/$N/match_$f.out | sed 's/ | ref.*//; s/^  . ours: //')
  mt=$(/usr/bin/grep "^  $f recall" $O/$N/match_$f.out | sed 's/^  . //; s/ ([0-9 /]* ref samples)//; s/ ([0-9 /]*)//')
  jd=$(/usr/bin/grep "^  $f:" $O/$N/judge_$f.out | sed -E 's/^  .: //; s/excursions [0-9]+ = [0-9.]+ \/m, //; s/ there-and-back [0-9.]+ \/m;//')
  [ -n "$lens" ] && [ -n "$mt" ] && [ -n "$jd" ] || { fail "empty match / judge fields ($O/$N/match_$f.out, judge_$f.out)"; continue; }
  echo "$N $f | $lens | $mt | $jd | $(/usr/bin/grep 'line ends' $O/$N/trace_$f.log | sed 's/.*kept lines): //')" >> $O/$N.sum.tmp
done
mv $O/$N.sum.tmp $O/$N.sum; cat $O/$N.sum; exit $rc

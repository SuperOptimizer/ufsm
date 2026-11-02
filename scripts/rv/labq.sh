#!/usr/bin/env bash
# scripts/rv/labq.sh [SCROLLS]: our-surface side labels for the rv model, one scroll at a time (scripts/v2/surflabels.sh: snapped wraps,
# strip.json trim, wrap-check dup/merged dropped, on-papyrus >= ONPAP, holdout boxes, plus the CT-coherence gate --min-coh COH),
# GRIDUP=1 (one CT level coarser: 8x smaller; the sampler interpolates), OUT /mnt/scratch/rv/labels (HDD; /vesuvius has no room).
# Paris 3 from COUNT's re-unroll (WSRC). Default list: every train scroll of scrolls.tsv with snapped wraps except Paris 4 (own GT).
# Env: U, TH (8), COH (gate), ONPAP (0.2), Q (queue log /vesuvius/ufsm/v2/rv/logs/labq.log). Skips scrolls with <OUT>/<S>/done.
set -u
H=$(cd "$(dirname "$0")/.." && pwd); V2=/vesuvius/ufsm/v2; OUT=${OUT:-/mnt/scratch/rv/labels}; Q=${Q:-/vesuvius/ufsm/v2/rv/logs/labq.log}
L=${1:-$(ls $V2/wraps_snap | grep -v -E '^(STOP|PHercParis4)$')}
for S in $L; do
  [ -e $OUT/$S/done ] && continue
  [ -f /vesuvius/ufsm/v2/rv/labq.stop ] && { echo "$(date) stop file" >> $Q; break; }
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); while [ "$avail" -lt 45 ]; do sleep 120; avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); done
  W=(); [ $S = PHercParis3 ] && W=(WSRC=$V2/count/unroll)
  echo "$(date) $S start" >> $Q
  env "${W[@]}" U=${U:-/vesuvius/ufsm/v2/rv/bin/ufsm} OUT=$OUT GRIDUP=1 TH=${TH:-8} ONPAP=${ONPAP:-0.2} bash $H/v2/surflabels.sh $S ${COH:+--min-coh $COH} --side-iso 64 >> $Q 2>&1
  echo "$(date) $S end: $(grep -o 'kept share [0-9.]*' $OUT/$S/surflabels.log | tail -1)" >> $Q
done

#!/usr/bin/env bash
# corpus.sh: every scroll of list.tsv in order through unroll.sh, one at a time (the whole desk per scroll). Restart-safe: finished scrolls
# (state/clean.done) and failed ones (state/failed: rm it to retry) are skipped. Before each scroll it waits until no other process holds
# a desk GPU (WAITGPU=0 skips). Stop: touch $C/STOP (between stages / slabs). Start on the desk:
#   cd /vesuvius/ufsm/corpus && setsid nohup bash src/scripts/corpus/corpus.sh > corpus.out 2>&1 < /dev/null &
# Env: ONLY ("PHerc1667 PHerc0139": just these, in this order) + everything unroll.sh takes.
set -u
H=$(cd "$(dirname "$0")" && pwd); C=${C:-/vesuvius/ufsm/corpus}
LIST=${ONLY:-$(tail -n +2 $C/list.tsv | cut -f1)}
for s in $LIST; do
  [ -e $C/STOP ] && break
  [ -e $C/$s/state/clean.done ] && continue
  [ -e $C/$s/state/failed ] && { echo "$s failed before, skipped"; continue; }
  while [ "${WAITGPU:-1}" = 1 ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
  cp $H/unroll.sh $H/.unroll.$s.sh; bash $H/.unroll.$s.sh $s   # (a copy: editing unroll.sh never touches a running scroll)
done
echo "$(date '+%m-%d %H:%M:%S') corpus.sh done" >> $C/corpus.log

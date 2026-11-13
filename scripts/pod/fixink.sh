#!/bin/bash
# fixink.sh S: a run whose ink stage started with the old latink.sh (grid-256 pass on 512 shards, ~7x windows): once its grid-0 pass is
# done (ink/lat.tmp/g0.done), stop that latink (and its grid-256 predict), clear the driver's ink.failed, rerun the current latink.sh
# (grid 0 kept, grid 256 with --shard 256, blends) and write state/ink.done so the driver's stripimg goes on.
S=${1:?}; O=/root/pod/unroll/$S; T=$O/ink/lat.tmp
log() { echo "$(date '+%F %T') fixink $S: $*" | tee -a $O/progress.log; }
until [ -e $T/g0.done ]; do pgrep -f "latink.sh $O " > /dev/null || { log "latink gone before grid 0 finished: nothing done"; exit 1; }; sleep 10; done
kt() { local c; for c in $(pgrep -P $1); do kt $c; done; kill $1 2> /dev/null; }
for p in $(pgrep -f "latink.sh $O "); do kt $p; done
for i in $(seq 1 30); do [ -e $O/state/ink.failed ] && break; sleep 1; done
rm -f $O/state/ink.failed; [ -n "$T" ] && rm -rf $T/g256
log "grid 0 done; the old grid-256 pass stopped; rerun latink (grid 256 on 256 shards)"
t0=$(date +%s)
if U=/root/pod/ufsm/build/ufsm ITH=${ITH:-13} bash $(dirname $(readlink -f $0))/latink.sh $O 0 $O/ink/inkpod-9000.zarr >> $O/logs/ink.log 2>&1; then
  echo "$(( $(date +%s) - t0 )) s wall (fixink: grid 256 rerun on 256 shards)" > $O/state/ink.done; log "ink done ($(( $(date +%s) - t0 )) s)"
else touch $O/state/ink.failed; log "ink rerun FAILED (logs/ink.log)"; fi

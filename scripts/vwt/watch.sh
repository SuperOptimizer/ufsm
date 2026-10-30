#!/usr/bin/env bash
# scripts/vwt/watch.sh [RUN...]: status of vwt training lanes on this host: last log line(s), lane events, checkpoints, GPU, memory.
#   KEEP=N: also copy every new checkpoint of each run to RUNROOT/RUN/keep/ (step-named) and keep the newest N there (0: off).
#   LOOP=S: repeat every S seconds (with KEEP: a snapshot keeper; stops when every run has a .stop file and no trainer is alive).
set -u
HOST=${HOST:-$(case $(hostname) in forlindesk2*) echo desk;; forlindesk*) echo forlindesk;; *) echo desk;; esac)}
RUNROOT=${RUNROOT:-/vesuvius/ufsm/vwt/runs}
runs=("$@"); [ ${#runs[@]} = 0 ] && runs=($(ls -d $RUNROOT/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null))
once() {
  echo "== $(date '+%m-%d %H:%M:%S') $HOST"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,power.limit --format=csv,noheader 2>/dev/null | sed 's/^/gpu /'
  free -g | awk '/Mem:/ {print "mem: used " $3 " GB, available " $7 " GB"}'
  for r in "${runs[@]}"; do
    local o=$RUNROOT/$r
    echo "-- $r$([ -e $o.stop ] && echo ' (stop file)')"
    [ -s $o.lane ] && tail -2 $o.lane | sed 's/^/  lane: /'
    [ -s $o.log ] && { [ -s $o/log.tsv ] && tail -1 $o/log.tsv; grep -E 'rror' $o.log; } | tail -${NLOG:-2} | cut -c1-220 | sed 's/^/  log: /'
    ls -t $o/*.vwt 2>/dev/null | head -3 | while read -r c; do echo "  ckpt: $(basename $c) $(date -r $c '+%m-%d %H:%M')"; done
    if [ "${KEEP:-0}" -gt 0 ]; then
      mkdir -p $o/keep
      for c in $o/step_*.vwt; do [ -e "$c" ] || continue; local t=$(date -r $c +%s); [ $(( $(date +%s) - t )) -lt 60 ] && continue   # (still being written)
        local k=$o/keep/$(basename ${c%.ckpt})-$t.ckpt; [ -e $k ] || cp $c $k; done
      ls -t $o/keep/*.vwt 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r c; do rm -f -- "$c"; done
    fi
  done
}
if [ -n "${LOOP:-}" ]; then
  while :; do once; alive=0; for r in "${runs[@]}"; do [ -e $RUNROOT/$r.stop ] || alive=1; done; [ $alive = 0 ] && exit 0; sleep $LOOP; done
else once; fi

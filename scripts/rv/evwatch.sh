#!/usr/bin/env bash
# scripts/rv/evwatch.sh RUN [EVERY 5000]: evaluate each new snapshot <run>/snap/<step>.ckpt (step % EVERY == 0) with scripts/rv/eval.sh.
# The eval needs the training GPU (one 16 GB card holds the trainer): PAUSE=1 (default) stops the trainer gracefully (STOP file + SIGTERM:
# it saves last.ckpt), runs the eval on that GPU, then relaunches it from <run>.relaunch (auto-resume). EVGPU / PAUSE=0: evaluate on another
# GPU (e.g. 0 when free) without pausing. Log <run>.evwatch.log; results /mnt/scratch/rv/eval/<name>-<step>.txt.
set -u
o=${1:?RUN}; EV=${2:-5000}; H=$(cd "$(dirname "$0")" && pwd); n=$(basename $o)
while :; do
  for c in $(ls $o/snap/*.ckpt 2>/dev/null | sort -V); do
    st=$(basename $c .ckpt); [ $((st % EV)) = 0 ] || continue
    [ -e $o/snap/$st.evaldone ] && continue
    echo "$(date) eval $st" >> $o.evwatch.log
    if [ "${PAUSE:-1}" = 1 ]; then
      touch $o.stop; [ -f $o.pid ] && kill -TERM $(cat $o.pid) 2>/dev/null
      while [ -f $o.launcher.pid ] && kill -0 $(cat $o.launcher.pid) 2>/dev/null; do sleep 10; done
      rm -f $o.stop
    fi
    GPU=${EVGPU:-$(sed -nE 's/.* GPU=([0-9]+) .*/\1/p' $o.relaunch)} bash $H/eval.sh $c $n-$st >> $o.evwatch.log 2>&1
    touch $o/snap/$st.evaldone
    if [ "${PAUSE:-1}" = 1 ]; then bash $o.relaunch; echo "$(date) relaunched" >> $o.evwatch.log; sleep 120; fi
  done
  [ -f $o.evwatch.stop ] && break
  sleep 300
done

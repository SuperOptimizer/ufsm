#!/usr/bin/env bash
# scripts/u0/watch.sh RUN_DIR...: keeps the desk trainers' speed while copy.sh downloads. Every 30 s it looks for NEW rows of each
# RUN_DIR/log.csv (ufsm train; a row per ~50 steps); baseline = the median samples_per_s of the last 12 rows at start. A new row below
# 0.97 x baseline or waiting on data (wait_s > 3 s per row; ~1 s blips occur without any copy) halves the copy rate cap in A/mbps (floor 5 MB/s); 6 healthy new rows in a row
# raise it 1.5x (up to MAX). A slow row also creates A/pause (prep.sh waits before each scan) until 3 healthy rows. Logs every change;
# exits with the STOP file or when the watched job (A/copy.pid: copy.sh or the prep run) is gone.
set -u
A=${A:-/vesuvius/ufsm/u0}; MAX=${MAX:-150}
declare -A base last
for r in "$@"; do
  base[$r]=$(tail -n 12 "$r/log.csv" | cut -d, -f11 | sort -n | awk '{a[NR] = $1} END{print a[int((NR + 1) / 2)]}')
  last[$r]=$(tail -n 1 "$r/log.csv" | cut -d, -f1)
  echo "$(date '+%m-%d %H:%M') baseline $(basename $r) ${base[$r]} samples/s (from step ${last[$r]})"
done
good=0
while [ ! -e $A/STOP ]; do
  [ -e $A/copy.pid ] && ! kill -0 $(cat $A/copy.pid) 2>/dev/null && { echo "$(date '+%m-%d %H:%M') copy finished: exit"; exit 0; }
  slow=""; fresh=0
  for r in "$@"; do
    IFS=, read -r step _ _ _ _ _ _ _ _ _ sps w < <(tail -n 1 "$r/log.csv")
    [ "$step" = "${last[$r]}" ] && continue
    last[$r]=$step; fresh=1
    awk -v s=$sps -v b=${base[$r]} -v w=$w 'BEGIN{exit !(s < 0.97 * b || w > 3)}' && slow="$slow $(basename $r)@$step $sps/${base[$r]} wait $w"
  done
  cur=$(cat $A/mbps)
  if [ -n "$slow" ]; then good=0; new=$(awk -v c=$cur 'BEGIN{n = c / 2; if (n < 5) n = 5; printf "%d", n}')
    echo $new > $A/mbps; touch $A/pause; echo "$(date '+%m-%d %H:%M') slow:$slow -> cap $new MB/s, prep paused"
  elif [ $fresh = 1 ]; then good=$((good + 1))
    [ $good -ge 3 ] && [ -e $A/pause ] && { rm -f $A/pause; echo "$(date '+%m-%d %H:%M') 3 healthy rows: prep resumed"; }
    if [ $good -ge 6 ] && [ $cur -lt $MAX ]; then new=$(awk -v c=$cur -v m=$MAX 'BEGIN{n = c * 1.5; if (n > m) n = m; printf "%d", n}'); echo $new > $A/mbps; good=0
      echo "$(date '+%m-%d %H:%M') 6 healthy rows -> cap $new MB/s"; fi
  fi
  sleep 30
done

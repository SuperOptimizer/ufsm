#!/usr/bin/env bash
# scripts/v2/snaploop.sh [NLANES]: repeat scrolls.sh (new CT / packed fields) + snap.sh until every train scroll of V2/scrolls.tsv has
# V2/wraps_snap/<S>/done, or STOP file V2/wraps_snap/STOP. Run: setsid nohup bash scripts/v2/snaploop.sh 3 > V2/logs/snaploop.out 2>&1 &
set -u
H=$(cd "$(dirname "$0")" && pwd); V2=${V2:-/vesuvius/ufsm/v2}
while [ ! -e $V2/wraps_snap/STOP ]; do
  bash $H/scrolls.sh > $V2/logs/scrolls.out 2>&1
  left=$(awk -F'\t' -v o=$V2/wraps_snap 'NR > 1 && $2 == "train" { if (system("test -e " o "/" $1 "/done")) n++ } END { print n + 0 }' $V2/scrolls.tsv)
  [ $left = 0 ] && { echo "$(date '+%F %T') snaploop: all train scrolls snapped"; break; }
  echo "$(date '+%F %T') snaploop: $left train scrolls left"
  bash $H/snap.sh ${1:-3}; sleep 300
done

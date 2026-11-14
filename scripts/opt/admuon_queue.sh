#!/bin/bash
# The adaptive Muon A/B queue on the perf pod (DESIGN.md 2026-11-14 "Fully adaptive Muon"): every arm takes /root/gpu.lock on its own,
# so other agents' runs interleave between arms. Usage (from the dir holding fx/ and klt1.lcb):
#   nohup ../scripts/opt/admuon_queue.sh ../build/ufsm ab 2000 [INIT_CKPT] > ab.queue.log 2>&1 &
set -u
BIN=$1 OUT=$2 STEPS=$3 INIT=${4:-}
D=$(cd "$(dirname "$0")" && pwd)
ARMS=${ARMS:-"muon:7 adaptive:7 normuon:7 lars:7 muon:8 dog:7"}   # arm:seed pairs
for a in $ARMS; do
  set -- ${a%%:*} ${a##*:}
  echo "$(date +%T) start $1 seed $2"
  timeout -k 10 7200 flock /root/gpu.lock timeout -k 10 6000 "$D/admuon_ab.sh" "$BIN" "$OUT" "$1" "$STEPS" "$2" "$INIT"
  echo "$(date +%T) done $1 seed $2: $(tail -1 "$OUT/r_$1_s$2.log")"
done

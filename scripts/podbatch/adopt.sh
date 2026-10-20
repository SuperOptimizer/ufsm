#!/usr/bin/env bash
# adopt.sh SCROLL SCAN UM LEVEL WINDOW: finish a predict whose gpu_lane worker was stopped (v5 -> v6 switch): wait for the predict pid in
# state/<s>.claim; on success the gpu_lane post-step (state predicted, reservations released, local CT deleted, up_one.sh under
# upload.lock), on failure the claim and reservations are released so a gpu_lane6 worker starts the volume again.
set -u; s=$1 scan=$2 um=$3 lv=$4 w=$5; cd /workspace; ST=out/STATUS.md
log() { ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) adopt $*" >> $ST ) 8> status.lock; }
read -r _ p _ < state/$s.claim; log "$s: waiting for predict pid $p (window $w)"
while kill -0 $p 2>/dev/null; do sleep 15; done
o=out/$s/${scan%%-*}-fitfields-20261016-ufsm-prod96-h2.zarr; tf=out/$s/time-$w.txt; sleep 2
if grep -q -E "non-zero|signal" $tf 2>/dev/null || [ ! -f $o/fields.json ]; then
  log "$s predict failed ($(head -1 $tf 2>/dev/null)): released for a new start"; rm -f state/$s.claim resv/$s gresv/$s; exit 1; fi
log "$s predicted at window $w, peak RSS $(( $(tail -1 $tf) / 1000000 )) GB"
touch state/$s.predicted; rm -f state/$s.claim resv/$s gresv/$s; rm -rf vol/$s scratch/$s-p
TAG=up flock -o upload.lock ./up_one.sh $s $scan $um $lv $w

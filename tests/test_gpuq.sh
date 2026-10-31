#!/usr/bin/env bash
# scripts/gpuq.sh on a private lock: a bg job yields to a later "now" job at its next box boundary, eval jobs are served before bg
# jobs that queued earlier, FIFO inside a priority, plain flock users stay mutually exclusive, dead tickets are dropped.
set -u; H=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf $T' EXIT; f=0
unset GPUQ_HELD GPUQ_FD GPUQ_OWNER GPUQ_PR GPUQ_SEQ GPUQ_TAG GPUQ_WAIT LOCK   # (run inside a held gpuq block, e.g. the suite's GPU block: no inherited grant)
export GPUQ_LOCK=$T/gpu.lock GPUQ_DIR=$T/q GPUQ_POLL=0.1
Q=$H/scripts/gpuq.sh; L=$T/order
bgjob() {   # 4 boxes of 1 s, yielding at each boundary
  . $Q; gpuq_acquire bg bgA; for b in 1 2 3 4; do echo "bgA$b" >> $L; sleep 1; gpuq_yield; done; gpuq_release; }
bgjob & p1=$!
until grep -q bgA2 $L 2> /dev/null; do sleep 0.02; done   # (box 2 started: not a fixed sleep, a loaded machine starts bgjob late)
bash $Q run now quick bash -c "echo now >> $L" &   # arrives during box 2: must run right after it
p2=$!
wait $p1 $p2
o=$(tr '\n' ' ' < $L); echo "yield order: $o"
[ "$o" = "bgA1 bgA2 now bgA3 bgA4 " ] || { echo "FAIL bg did not yield to now at a box boundary"; f=1; }
# a holder; then bg, eval, eval, now queue in that order -> served now, eval1, eval2, bg
: > $L
bash $Q run eval hold bash -c "sleep 1.2; echo hold >> $L" & ph=$!; sleep 0.3
bash $Q run bg b bash -c "echo bg >> $L" & sleep 0.2
bash $Q run eval e1 bash -c "echo e1 >> $L" & sleep 0.2
bash $Q run eval e2 bash -c "echo e2 >> $L" & sleep 0.2
bash $Q run now n bash -c "echo now >> $L" &
wait
o=$(tr '\n' ' ' < $L); echo "priority order: $o"
[ "$o" = "hold now e1 e2 bg " ] || { echo "FAIL priority / FIFO order"; f=1; }
# plain flock users and queue jobs exclude each other
: > $L
flock $GPUQ_LOCK bash -c "echo p0 >> $L; sleep 0.8; echo p1 >> $L" & sleep 0.2
bash $Q run now n bash -c "echo q >> $L"; wait
o=$(tr '\n' ' ' < $L); [ "$o" = "p0 p1 q " ] || { echo "FAIL plain flock exclusion: $o"; f=1; }
# a dead ticket does not block the queue
mkdir -p $GPUQ_DIR; echo ghost > $GPUQ_DIR/t.0.000000001.999999
timeout 5 bash $Q run bg x true || { echo "FAIL dead ticket blocked"; f=1; }
# nested: a command run under the lock acquires again at once
timeout 5 bash $Q run eval outer bash $Q run now inner true || { echo "FAIL nested acquire"; f=1; }
bash $Q status | head -3
[ $f = 0 ] && echo "gpuq ok"; exit $f

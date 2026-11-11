#!/bin/bash
# fibhv-d (training pod): the fibhv-c recipe (H / V masks + direction head, train --fib-dir 1) from fibhv-c's latest snapshot, on every
# published fibre set (scripts/fibers/mksources3.py; labels: labels2 = mklabels2.sh, /workspace/fibd/labels = mklabels3.sh + the spiral
# raster); snapshots every 3000 (evals at 3000 and every 6000: scripts/fibers/evald_desk.sh). INIT = the fibhv-c snapshot.
set -u
cd /workspace/ufsm-fibd
INIT=${INIT:?fibhv-c snapshot}
CTL=/workspace/ct/volcomp CTM=https://dl.ash2txt.org/community-uploads/forrest/volcomp L2=/workspace/fibers/labels2 L3=/workspace/fibd/labels \
  META=/workspace/fibers/meta CACHE=/workspace/cache python3 -I scripts/fibers/mksources3.py /workspace/fibd/sources-fibhv-d.json
R=/workspace/fibers/runs SRCJ=/workspace/fibd/sources-fibhv-d.json U=/workspace/ufsm-fibd/build/ufsm GPU=0 LOCK= MEMCAP= P=512 WORKERS=8 SLOTS=10 MINMEM=16 VALB=0 SNAP=3000 \
  EXTRA="--fib-dir 1 --init-from $INIT" \
  nohup bash scripts/fibers/train_hv.sh fibhv-d ${STEPS:-30000} > /workspace/fibers/runs/fibhv-d.out 2>&1 < /dev/null &
echo launched

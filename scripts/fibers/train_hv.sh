#!/usr/bin/env bash
# scripts/fibers/train_hv.sh NAME STEPS: the H / V fibre mask model from scratch: the default mask task (2 outputs: channel 0 "recto"
# = H fibres, channel 1 "sheet" = V fibres, BCE + dice over the known voxels) on the rasterised published fibre tracings of PHerc1447
# and PHerc0175A at 8.64 um (scripts/fibers/mklabels.sh, mksources.sh). It is scripts/rv/train.sh with other heads and sources: same
# 7-level net, MX-fp4, --mem auto, --gpu-aug 2, auto-resume, STOP file, snapshots. GPU / LOCK / P / EXTRA as there.
# NOT started by the fibre work (2026-11-10: both GPUs in use); the first launch should be a 200-step smoke run (UFSM_MEM_REPORT=1).
set -u
D=$(dirname "$(readlink -f "$0")")
export R=${R:-/vesuvius/ufsm/fibers/runs} SRCJ=${SRCJ:-/vesuvius/ufsm/fibers/sources-fibhv.json} U=${U:-/home/forrest/ufsm-fib/build/ufsm}
export HEADS=${HEADS:-"--dice 0.5 --pos-weight 4"} P=${P:-512}
mkdir -p $R
# the trainer in the training memory slice (MEMCAP, default "-t -m 40G"; "" = none): a wrapper, since rv/train.sh runs "$U train ..."
if [ -n "${MEMCAP--t -m 40G}" ]; then
  W=$R/ufsm-memcap.sh; printf '#!/bin/bash\nexec ~/bin/memcap %s %q "$@"\n' "${MEMCAP--t -m 40G}" "$U" > $W; chmod +x $W; export U=$W
fi
exec bash $D/../rv/train.sh "$@"

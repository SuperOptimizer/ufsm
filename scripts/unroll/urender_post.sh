#!/bin/bash
# scripts/unroll/urender_post.sh [CKPT] [TAG]: the post-mega GPU re-render of the urender audit (2026-11-14) on PHerc1667-u1, then the
# CPU checks against the published unwrap (scripts/unroll/pubcmp.py). Queue AFTER the mega test pass (GPU 0 via u1667ink.sh's
# flock /tmp/gpu0-bench.lock); not before.
#   1. u1667ink.sh with GATE=1 (ink + c12 / c13, two grids + ink-blend, --axis) and the fixed render recipe: --fill-ct 80 (recentre CT + ink
#      on the nearest papyrus within 4 voxels; was 1 = only masked-zero CT void: ~50 % of the 1667 surface sampled air), --fill-pm 0
#   2. pubcmp.py on the new renders, ungated and papyrus-gated: direction (sawtooth) + AUC vs the published ink; gate line in the log
#   3. the same pubcmp on the old renders (inkr/inkpod-9000) when not done yet: the before row
# Expected (urender.md): CT < 80 on flag-0 pixels 45-54 % -> about the gap share; ink AUC vs published 0.51 before. A dir +1 strip
# fails the sawtooth whatever the ink (re-run on fibfit's dir -1 strip: STRIP=<its flat dir>).
# Env: SCROLL (PHerc1667), U (ufsm), STRIP (flat dir; default the run's strip/strip/flat), PUB, BANNER, OUTQ (results dir).
set -u
CK=${1:-/vesuvius/ufsm/v2/lat/runs/inkpod/snap/15000.ckpt}; TAG=${2:-inkpod-$(basename $CK .ckpt)-fct80}; SCROLL=${SCROLL:-PHerc1667}
O=/vesuvius/ufsm/unroll/$SCROLL-u1; HERE=$(dirname $(readlink -f $0)); STRIP=${STRIP:-$O/strip/strip/flat}
PUB=${PUB:-/vesuvius/ufsm/gt/pub1667-unwrap}; BANNER=${BANNER:-/vesuvius/ufsm/previews-20261114/1667full/published/banner-full.webp}
OUTQ=${OUTQ:-/vesuvius/ufsm/audit-20261114/urender/post}; mkdir -p $OUTQ
log() { echo "$(date '+%F %T') urender_post $SCROLL $TAG: $*" | tee -a $OUTQ/run.log; }
[ -s $CK ] || { log "no checkpoint $CK"; exit 1; }
log "render (GATE=1, --fill-ct 80, --fill-pm 0)"
GATE=1 SCROLL=$SCROLL FILLX="--fill 4 --fill-ct 80 --fill-grid 50000 --fill-inpaint 4 --fill-pm 0" bash $HERE/../inkgt/u1667ink.sh $CK $TAG ||
  { log "u1667ink failed"; exit 1; }
RD=$O/inkr/$TAG
pc() {   # NAME RENDER-DIR [GATE-DIRS]
  [ -s $OUTQ/$1/summary.json ] && { log "$1: done before"; return 0; }
  CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m 30G python3 -I $HERE/pubcmp.py --flat $STRIP --pub $PUB --banner $BANNER --out $OUTQ/$1 \
    --renders $2 ${3:+--ink-gate $3} > $OUTQ/$1.log 2>&1; log "$1: $(grep '^pubcmp:' $OUTQ/$1.log)"; }
pc before $O/inkr/inkpod-9000
pc after $RD
[ -d $RD/c12 ] && pc after_gated $RD $RD/c12,$RD/c13
log "done: $OUTQ/{before,after,after_gated}/summary.json"

#!/usr/bin/env bash
# winsweep.sh [ARMS..]: inference window size vs unroll quality (lead 2026-11-20, netsplit DESIGN). wind96 (full-res CT net) fit fields predicted
# at cube windows N^3 (halo 64) on the wgate box set (44 core boxes) + the 4 Paris 4 slab sites, each arm a wgate run (wgate.sh: predict ->
# box fit + 2 shifted sub-domain fits -> GT ids; slabs -> fit-join -> joined ids) scored against the cached wind96 rows, which are the
# production 640 x 640 x 384 arm (fits deterministic: wgate never re-runs them). Paired per box with bootstrap CIs: wbench-stats.
#   ARMS: names wN (cube N, default "w256 w384 w512"); wNxZ = N x N x Z (e.g. w640x640: the production xy at a cube).
# GPU in short windows: BUDGET seconds per invocation (default 540, inside a <= 10 min card-1 slot); wgate skips boxes / sites that
# already have rows, so rerun this script in the next slot until every arm prints its table. Env: GPU card (CUDA_VISIBLE_DEVICES, 1),
# SITE (desk), CK (/vesuvius/ufsm/inksdk/ckw/wind96-u2.ckpt), SITES (wgate's 4 sites; "" for boxes only), BOXES (core).
set -u
H=$(cd "$(dirname "$0")" && pwd)
CK=${CK:-/vesuvius/ufsm/inksdk/ckw/wind96-u2.ckpt}; BUDGET=${BUDGET:-540}; export SITE=${SITE:-desk}
[ -e "$CK" ] || { echo "winsweep: no checkpoint $CK"; exit 2; }
ARMS=${*:-w256 w384 w512}; t0=$(date +%s)
for a in $ARMS; do
  n=${a#w}; xy=${n%x*}; z=${n#*x}; [ "$z" = "$n" ] && z=$xy
  left=$(( BUDGET - ($(date +%s) - t0) )); [ $left -gt 30 ] || { echo "winsweep: budget spent before $a (rerun in the next slot)"; break; }
  echo "winsweep: $a (window $xy x $xy x $z, halo 64), ${left} s left"
  # (GPU 0 inside the visible card; NOLOCK=0: wgate's own per-box GPU lock serialises with other wbench users)
  # (its own process group: at the budget the whole wgate tree (GPU lane, box lanes, slab lane) stops, not only the top shell)
  CUDA_VISIBLE_DEVICES=${GPU:-1} GPU=0 WIN=$xy HALO=64 WINZ=$z HALOZ=64 BOXES=${BOXES:-core} SITES="${SITES-4624 5392 5904 6672}" \
    setsid bash $H/wgate.sh "$CK" win-$a & pg=$!
  while kill -0 $pg 2> /dev/null && [ $(( $(date +%s) - t0 )) -lt $BUDGET ]; do sleep 5; done
  if kill -0 $pg 2> /dev/null; then kill -TERM -- -$pg; sleep 20; kill -KILL -- -$pg 2> /dev/null; echo "winsweep: $a stopped by the budget (rerun resumes it)"; break; fi
  wait $pg
done
for a in $ARMS; do t=${WB:-/vesuvius/ufsm/wbench}/gate/win-$a/table.txt; [ -s $t ] && { echo "== $a"; cat $t; }; done

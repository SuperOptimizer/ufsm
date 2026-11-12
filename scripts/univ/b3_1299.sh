#!/usr/bin/env bash
# scripts/univ/b3_1299.sh: B3 PHerc1299 (2026-11-11). PHerc1299 is a v2 whole-scroll holdout (split holdout), so snap.sh skips it: snap its
# wraps here (scripts/unroll/snap_existing.sh, target mix, as snap.sh would), then the good.sh labels + gate with HO = the wbench boxes plus
# the held-out bottom third (scripts/v2/holdband.tsv; surf-labels ignores the band + 0.5 mm, and the training sources hold it out again).
# Waits for the PHerc1218 labels first (the dev memory slice: one ~30 GB job at a time).
set -u; H=$(cd "$(dirname "$0")/../.." && pwd); V2=/vesuvius/ufsm/v2; U=${U:-$H/build/ufsm}; B3=/mnt/scratch/univ/b3; S=PHerc1299
until [ -e $B3/good/PHerc1218/done ] || ! pgrep -f "rv/good[.]sh PHerc1218" > /dev/null; do sleep 120; done
if [ ! -e $V2/wraps_snap/$S/done ]; then
  mkdir -p $V2/wraps_snap/$S
  RUN=$V2/wraps_src/$S CTROOT=$V2/ct SNAPT=mix MEASURE=0 TH=${TH:-10} U=$U nice -n 10 bash $H/scripts/unroll/snap_existing.sh $S $V2/wraps_snap > $V2/wraps_snap/$S/driver.log 2>&1 \
    && [ $(ls $V2/wraps_snap/$S/wraps_snap | grep -c '\.sfc$') = $(ls $V2/wraps_src/$S/wraps | grep -c '\.sfc$') ] && touch $V2/wraps_snap/$S/done \
    || { echo "$(date) $S snap failed ($V2/wraps_snap/$S/driver.log)"; exit 1; }
fi
HO=/vesuvius/ufsm/univ/holdouts-1299.json OUT=$B3/good LAB=$B3/labels Q=$B3/good.log U=$U TH=${TH:-10} KEEPFULL=1 nice -n 5 bash $H/scripts/rv/good.sh $S

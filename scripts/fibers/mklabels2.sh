#!/usr/bin/env bash
# scripts/fibers/mklabels2.sh [THREADS]: H / V fibre labels with a negative ring (fiber-raster --ring), for fibhv-b. As mklabels.sh
# (prob254, level-0 8.64 um, air from the CT) plus: radius 2, guard 2 (unknown 2..4 voxels), ring 3 (0 = "no fibre axis" at 4..7
# voxels, 35..60 um), vetoed within 4 voxels of any fibre of the all-lengths tracing (0175A_SMALLMORE.afv, PHerc1447_all_v2.afv).
# Ring width from the measured axis spacing (DESIGN.md 2026-11-11): nearest same-family axis in the > 1 mm tracings median 14.7 voxels,
# 5th percentile 9; the ring ends at 7. Output /vesuvius/ufsm/fibers/labels2/<scroll>/{H,V}/8.64. CPU.
set -u
U=${U:-/home/forrest/ufsm-fib/build/ufsm}; A=/vesuvius/ufsm/fibers/raw; O=${O:-/vesuvius/ufsm/fibers/labels2}; CT=/vesuvius/ufsm/v2/ct; T=${1:-3}
mkdir -p $O
run() {   # run SCROLL CTKEY ZBAND EXTRA
  [ -e $O/$1/done ] && { echo "$1: done"; return; }
  until [ -s $A/$4.check ] && grep -q OK $A/$4.check; do sleep 60; done   # (the all-lengths AFV downloaded and verified)
  nice -n 19 ~/bin/memcap -m 16G $U fiber-raster $A/$1_1mm_v2.afv $O/$1 --um 8.64 --ct $CT $2/0 --iso 64 --level 0 --radius 2 --guard 2 --ring 3 \
    --extra $A/$4 --zband $3 --threads $T > $O/$1.log 2>&1 && touch $O/$1/done
  tail -1 $O/$1.log
}
run PHerc0175A PHerc0175A/20250521115057-8.640um-1.2m-116keV-masked.zarr 1024,11776 0175A_SMALLMORE.afv &
run PHerc1447 PHerc1447/20250521151220-8.640um-1.2m-116keV-masked.zarr 0,24576 PHerc1447_all_v2.afv &
wait

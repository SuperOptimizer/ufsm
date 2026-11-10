#!/usr/bin/env bash
# scripts/fibers/mklabels.sh [THREADS]: H / V fibre mask labels (ufsm fiber-raster, prob254 on the level-0 CT grid, air from the CT)
# for the two scrolls whose published automated fibre volumes are native to our scans: PHerc1447 and PHerc0175A (8.64 um).
# AFVs: /vesuvius/ufsm/fibers/raw (community upload AutomatedFiberVolumes, *_1mm_v2.afv). Output /vesuvius/ufsm/fibers/labels/<scroll>/{H,V}/8.64.
# CPU only; a few hours at 2 threads per scroll. DESIGN.md 2026-11-10.
set -u
U=${U:-/home/forrest/ufsm-fib/build/ufsm}; A=/vesuvius/ufsm/fibers/raw; O=${O:-/vesuvius/ufsm/fibers/labels}; CT=/vesuvius/ufsm/v2/ct; T=${1:-2}
mkdir -p $O
run() {   # run SCROLL CTKEY ZBAND
  [ -e $O/$1/done ] && { echo "$1: done"; return; }
  nice -n 19 ~/bin/memcap -m 16G $U fiber-raster $A/$1_1mm_v2.afv $O/$1 --um 8.64 --ct $CT $2/0 --iso 64 --level 0 --radius 2 --guard 3 --zband $3 --threads $T > $O/$1.log 2>&1 && touch $O/$1/done
  tail -1 $O/$1.log
}
run PHerc0175A PHerc0175A/20250521115057-8.640um-1.2m-116keV-masked.zarr 1024,11776 &
run PHerc1447 PHerc1447/20250521151220-8.640um-1.2m-116keV-masked.zarr 0,24576 &
wait

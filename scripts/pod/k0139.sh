#!/usr/bin/env bash
# k0139.sh K: PHerc0139 winding labels at the scan's native 9.362 um (level 0) = the desk's v2/rv/gt/p0139full/build.sh: codes A / B
# rastered from the q-assigned wrap surfaces qA / qB (K/gt/p0139-q: copied from the desk's gt/PHerc0139-wind-20261016, 55 MB; their 10-16
# wind-seed / wind-assign predates the 10-19 sense rule, so a fresh wind-seed is not guaranteed to reproduce them), pair-rule phase per set
# (radius 768 um = 82 voxels), fused (tol 0.1).  Output K/gt/p0139full/phase/9.362.
set -euo pipefail; K=$1; U=${U:-/workspace/lat/build/ufsm}; T=${T:-6}
Q=$K/gt/p0139-q; O=$K/gt/p0139full; AX=$K/small/axis/PHerc0139-umbilicus.json; SH=20974,6621,6621; mkdir -p $O
key() { ls $1 | grep -v json | sort -g | head -1; }
for s in A B; do
  [ -f $O/codes$s/zarr.json ] || nice -n 10 $U raster $O/codes$s --shape $SH --um 9.362 --level 0 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $Q/q$s/*.tifxyz 2>&1 | tail -1
  [ -f $O/phase$s/zarr.json ] || nice -n 10 $U phase-raster $O/codes$s $(key $O/codes$s) $O/phase$s --gs 2 --pair 1 --radius 82 --axis $AX --threads $T --shards 1 2>&1 | tail -1
done
[ -f $O/phase/zarr.json ] || nice -n 10 $U phase-fuse $O/phase 9.362 $O/phaseA $(key $O/phaseA) $O/phaseB $(key $O/phaseB) --tol 0.1 --threads $T
echo done

#!/usr/bin/env bash
# p4kit.sh K: Paris 4 winding labels on a pod from the published June / July segments (scripts/pod/fetchsegs.sh PHercParis4 K/seg/PHercParis4):
#   June -> wind-seed (set A) -> raster codesA 9.6 (level 2) -> 19.2 max reference -> July wind-assign (set B) -> raster codesB 9.6
#   (= scripts/wbench/gtjuly/build.sh, which reproduced the 10-19 desk seed / assign identically), pair-rule phase A / B at 9.6 with the desk
#   p4full flags (gs 4 on codes already on the 9.6 grid, radius 160, the p4full shard box), and the wbench paris4 codes (q-9p6: codesA rastered
#   at 4.8 um, level 1, then halved with max).  Outputs K/gt/paris4/{codesA,codesB,phaseA,phaseB,q-9p6}.  Restartable (each step skips done output).
set -uo pipefail; K=$1; U=${U:-/workspace/lat/build/ufsm}; T=${T:-10}
SEG=$K/seg/PHercParis4; G=$K/gt/paris4; AX=$K/small/axis/paris4-axis-0.json; SH=75784,32693,32693; UM=2.4; mkdir -p $G
step() { echo "== $(date +%T) $*"; }
A=$(ls -d $SEG/20260623*-w*/mesh/*-on-20260411134726-2.4um.tifxyz | grep -v jordi); B=$(ls -d $SEG/20260701*/mesh/*-on-20260411134726-2.4um.tifxyz)
echo "June $(echo "$A" | grep -c .) July $(echo "$B" | grep -c .) surfaces"
step seed June; [ -s $G/qA/seed.jsonl ] || nice -n 5 $U wind-seed $G/qA $A --vol-um $UM --axis $AX > $G/seedA.log 2>&1 || { tail -3 $G/seedA.log; exit 1; }
step raster June 9.6; [ -f $G/codesA/zarr.json ] || nice -n 5 $U raster $G/codesA --shape $SH --um $UM --level 2 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qA/*.tifxyz 2>&1 | tail -1
step ref 19.2; [ -d $G/ref ] || nice -n 5 $U downsample $G/codesA 9.6 $G/ref 19.2 --mode max --threads $T 2>&1 | tail -1
step assign July; [ -s $G/qB/assign.jsonl ] || nice -n 5 $U wind-assign $G/ref 19.2 $G/qB $B --ref-um 19.2 --vol-um $UM --axis $AX --tol 2 --step 4 --threads $T > $G/assignB.log 2>&1 || { tail -3 $G/assignB.log; exit 1; }
step raster July 9.6; [ -f $G/codesB/zarr.json ] || nice -n 5 $U raster $G/codesB --shape $SH --um $UM --level 2 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qB/*.tifxyz 2>&1 | tail -1
step wbench codes 4.8 -> q-9p6; [ -f $G/codesA48/zarr.json ] || nice -n 5 $U raster $G/codesA48 --shape $SH --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qA/*.tifxyz 2>&1 | tail -1
[ -d $G/q-9p6.zarr/9.6 ] || nice -n 5 $U downsample $G/codesA48 4.8 $G/q-9p6.zarr 9.6 --mode max --threads $T 2>&1 | tail -1
for s in A B; do step phase $s; [ -f $G/phase$s/zarr.json ] && [ ! -e $G/phase$s.partial ] || { touch $G/phase$s.partial
  nice -n 5 $U phase-raster $G/codes$s 9.6 $G/phase$s --gs 4 --codes-grid 1 --pair 1 --radius 160 --axis $AX --threads $T --shards 1 --resume 1 \
    --box 26624,4096,4096,43008,26624,26624 2>&1 | tail -2 && rm -f $G/phase$s.partial; }; done
step done; du -sh $G/*

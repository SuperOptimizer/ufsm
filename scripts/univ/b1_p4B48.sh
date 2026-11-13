#!/usr/bin/env bash
# scripts/univ/b1_p4B48.sh (desk): the Paris 4 July codes at level 1 (4.8 um) for the onfly phase (PHercParis4-july48 in mksources_univ.py),
# from the July meshes still on the desk (wbench/gtjuly/aws: 2026-11-11, no re-download needed): the 19.2 reference from the June 9.6 codes
# (gtjuly/codesA, downsample --mode max), wind-assign of the July meshes in that gauge (the gtjuly build.sh flags; assign.jsonl checked
# against gtjuly/qB and the 10-19 GT), then raster --level 1. Runs after B3's PHerc1299 chain (the dev memory slice). Out GN/codesB48
# on NVMe (2026-11-12: /mnt/scratch is the live trainer's I/O bottleneck, so the 43 GB of new shards do not go there).
set -u; H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}; T=${T:-8}
J=/vesuvius/ufsm/wbench/gtjuly; G=/mnt/scratch/univ/b1/gt/paris4; GN=${GN:-/vesuvius/ufsm/univ/b1/gt/paris4}; SH=75784,32693,32693; UM=2.4; AX=/vesuvius/ufsm/teacher/axis-0.json
step() { echo "== $(date +%T) $*"; }
until ! pgrep -f "univ/b3_1299[.]sh" > /dev/null; do sleep 120; done
B=$(ls -d $J/aws/PHercParis4/segments/20260701*/mesh/*-on-20260411134726-2.4um.tifxyz)
step ref 19.2; [ -e $G/ref19/zarr.json ] || nice -n 5 $U downsample $J/codesA 9.6 $G/ref19 19.2 --mode max --threads $T 2>&1 | tail -1
step assign July; [ -s $G/qB/assign.jsonl ] || nice -n 5 $U wind-assign $G/ref19 19.2 $G/qB $B --ref-um 19.2 --vol-um $UM --axis $AX --tol 2 --step 4 --threads $T > $G/assignB.log 2>&1
for ref in $J/qB/assign.jsonl /vesuvius/ufsm/gt/paris4-wind-20261019/qB/assign.jsonl; do
  cmp -s <(jq -S -c . $G/qB/assign.jsonl | cut -c1-200) <(jq -S -c . $ref | cut -c1-200) && echo "assign identical to $ref" || echo "assign DIFFERS from $ref"; done
step raster July 4.8; mkdir -p $GN; [ -f $GN/codesB48/done ] || { nice -n 5 $U raster $GN/codesB48 --shape $SH --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qB/*.tifxyz > $GN/rasterB48.log 2>&1 && touch $GN/codesB48/done; tail -c 120 $GN/rasterB48.log; }
step done

#!/usr/bin/env bash
# scripts/univ/b1_p4B48.sh (desk): the Paris 4 July codes at level 1 (4.8 um) for the onfly phase (PHercParis4-july48 in mksources_univ.py),
# from the July meshes still on the desk (wbench/gtjuly/aws: 2026-11-11, no re-download needed): the 19.2 reference from the June 9.6 codes
# (gtjuly/codesA, downsample --mode max), wind-assign of the July meshes in that gauge (the gtjuly build.sh flags; assign.jsonl checked
# against gtjuly/qB and the 10-19 GT), then raster --level 1. Runs after B3's PHerc1299 chain (the dev memory slice). Out GN/codesB48
# on NVMe (2026-11-12: /mnt/scratch is the live trainer's I/O bottleneck, so the 43 GB of new shards do not go there).
set -uo pipefail; H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}; T=${T:-8}
# (audit mini/shell #6 / #7 / #10: every step's status is fatal, a failed assign never reaches the raster, the skips test done markers
# written on success (not the output's existence, which a killed run leaves), the paths take env overrides)
J=${J:-/vesuvius/ufsm/wbench/gtjuly}; G=${G:-/mnt/scratch/univ/b1/gt/paris4}; GN=${GN:-/vesuvius/ufsm/univ/b1/gt/paris4}; SH=75784,32693,32693; UM=2.4
AX=${AX:-/vesuvius/ufsm/teacher/axis-0.json}; GT1019=${GT1019:-/vesuvius/ufsm/gt/paris4-wind-20261019}
step() { echo "== $(date +%T) $*"; }
die() { echo "b1_p4B48: $*" >&2; exit 1; }
until ! pgrep -f "univ/b3_1299[.]sh" > /dev/null; do sleep 120; done
B=$(ls -d $J/aws/PHercParis4/segments/20260701*/mesh/*-on-20260411134726-2.4um.tifxyz) || die "no July mesh under $J/aws"
step ref 19.2; [ -e $G/ref19.done ] || { rm -rf $G/ref19; nice -n 5 $U downsample $J/codesA 9.6 $G/ref19 19.2 --mode max --threads $T > $G/ref19.log 2>&1 && touch $G/ref19.done; } || die "downsample failed ($G/ref19.log)"
tail -1 $G/ref19.log 2>/dev/null
step assign July; [ -e $G/qB.done ] || { nice -n 5 $U wind-assign $G/ref19 19.2 $G/qB $B --ref-um 19.2 --vol-um $UM --axis $AX --tol 2 --step 4 --threads $T > $G/assignB.log 2>&1 && touch $G/qB.done; } || die "wind-assign failed ($G/assignB.log)"
same=0
for ref in $J/qB/assign.jsonl $GT1019/qB/assign.jsonl; do
  cmp -s <(jq -S -c . $G/qB/assign.jsonl | cut -c1-200) <(jq -S -c . $ref | cut -c1-200) && { echo "assign identical to $ref"; same=1; } || echo "assign DIFFERS from $ref"; done
[ $same = 1 ] || die "assign matches neither reference: not rastering (rm $G/qB.done to redo the assign)"
step raster July 4.8; mkdir -p $GN; [ -f $GN/codesB48/done ] || { nice -n 5 $U raster $GN/codesB48 --shape $SH --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qB/*.tifxyz > $GN/rasterB48.log 2>&1 && touch $GN/codesB48/done; rc=$?; tail -c 120 $GN/rasterB48.log; [ $rc = 0 ]; } || die "raster failed ($GN/rasterB48.log)"
step done

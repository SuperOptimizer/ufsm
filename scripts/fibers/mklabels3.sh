#!/usr/bin/env bash
# scripts/fibers/mklabels3.sh [THREADS]: the fibhv-d label rasters that labels2 (mklabels2.sh: PHerc1447, PHerc0175A at 8.64 um) does
# not cover, with the same ring recipe (radius 2, guard 2, ring 3; prob254 H / V planes; DESIGN.md 2026-11-11 "fibhv-d"):
#   PHercParis4-l2   Paris4_1mm_v2 (2023 7.91 um frame) through the ufsm reg map onto our 2.4 um scan, level 2 (9.6 um), z band
#                    6656..11776 (level-2 voxels: the 2023 scan's extent); the ring is vetoed near any fibre of Paris4_all_v2 (mapped)
#                    or Paris4_v1 (native 2.4 um chains, no families).
#   PHerc0125        PHerc0125_3mm_v2 (native 9.362 um, level 0), z 9216..12288; no all-lengths tracing exists for 0125, so no veto.
#   hf-<scroll>      the public human fibre lines (tools/hfjson.py -> UFIB) of PHerc0139, PHerc0332, PHerc1451, PHercParis4 on their
#                    2.4 um scans at level 2 (9.596 / 9.6 um); no veto.
# No --ct: no air negatives (the sampler draws windows around papyrus inside stored label chunks, fibnet 3d84ada), so no CT is read.
# Pod: A=/workspace/fibd/raw H=/workspace/fibd/hf M=/workspace/fibd/meta/reg.json O=/workspace/fibd/labels U=/workspace/ufsm-fibd/build/ufsm
set -u
U=${U:-/home/forrest/ufsm-fibd/build/ufsm}; A=${A:-/vesuvius/ufsm/fibers/raw}; H=${H:-/mnt/scratch/fibd/hf}; O=${O:-/mnt/scratch/fibd/labels}
M=${M:-/vesuvius/ufsm/reg/PParis4_2026to2023/reg.json}; T=${1:-8}; mkdir -p $O
R="--radius 2 --guard 2 --ring 3 --resume 1 --threads $T"
run() {   # run NAME INPUT ARGS...
  local n=$1 in=$2; shift 2; [ -e $O/$n/done ] && { echo "$n: done"; return; }
  until [ -s $in ] && { [ "${in##*.}" = fib ] || [ -s $A/$(basename $in).ok ]; }; do sleep 60; done
  nice -n 10 $U fiber-raster $in $O/$n "$@" $R > $O/$n.log 2>&1 && touch $O/$n/done
  echo "$n: $(tail -1 $O/$n.log)"
}
X=$A/Paris4_all_v2.afv@map,$A/Paris4_v1.afv
until [ -s $A/Paris4_all_v2.afv.ok ]; do sleep 60; done
run PHercParis4-l2 $A/Paris4_1mm_v2.afv --um 9.6 --shape 18946,8174,8174 --level 2 --map $M --zband 6656,11776 --extra $X
run PHerc0125 $A/PHerc0125_3mm_v2.afv --um 9.362 --shape 20840,8387,8387 --level 0 --zband 9216,12288
run hf-PHerc0139 $H/PHerc0139.fib --um 9.596 --shape 19239,6628,6628 --level 2
run hf-PHerc0332 $H/PHerc0332.fib --um 9.596 --shape 8398,3941,3941 --level 2
run hf-PHerc1451 $H/PHerc1451.fib --um 9.596 --shape 14986,5203,5203 --level 2
run hf-PHercParis4 $H/PHercParis4.fib --um 9.6 --shape 18946,8174,8174 --level 2
echo "all done"

#!/bin/bash
# axcheck.sh S [AXIS]: the scroll's axis (default: the kit's corpus axis) vs its papyrus centroid on level LV+3 (~70-77 um).
# Exit 0 ok, 3 off (/root/pod/scrolls/S/axis_centroid.json written), other = could not check. One line -> /root/pod/axcheck.log
S=${1:?}; POD=${POD:-/root/pod}; HERE=$(dirname $(readlink -f $0)); . $POD/scrolls/$S.env; AX=${2:-$AXS}; L=$((LV + 3)); FF=$((1 << L))
read -r nz ny nx < <($U info $R $K/$L 2>/dev/null | awk '/shape/ {print $2, $4, $6; exit}')
[ -n "$nx" ] || { echo "$(date '+%F %T') $S axcheck: no level $L" | tee -a $POD/axcheck.log; exit 2; }
T=$POD/ax.$S.raw; UFSM_NCPU=4 $U read $R $K/$L 0 0 0 $nz $ny $nx $T --threads 4 > /dev/null 2>&1 || exit 2
um0=$UM   # (list.tsv um = the level-0 voxel um; the axis files are in level-0 voxels)
o=$(python3 -I $HERE/axcheck.py $T $nz $ny $nx $FF $um0 $AX $POD/scrolls/$S/axis_centroid.json); rc=$?; rm -f $T
echo "$(date '+%F %T') $S (L$L, $nz x $ny x $nx, axis $AX): $o" | tee -a $POD/axcheck.log; exit $rc

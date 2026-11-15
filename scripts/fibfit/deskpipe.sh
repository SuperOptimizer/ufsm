#!/bin/bash
# deskpipe.sh OUT REF [extra fit flags]: 1667 slab fits on the desk CPU (unroll.sh lane recipe: I 64, OV 16, --crop-material 2), fit-join,
# wraps, wrap-snap (mix), wrap-dump (every 4) -> OUT/ours4.f32. Run inside memcap + lowprio.
O=$1; REF=$2; shift 2; X="$*"; U=/home/forrest/ufsm-fibfit/build/ufsm; F=/vesuvius/ufsm/unroll/PHerc1667-u1/fields/lat; S=/vesuvius/ufsm/unroll/PHerc1667-u1
export CUDA_VISIBLE_DEVICES= UFSM_FIT_THREADS=8
mkdir -p $O/slabs; ze=$(jq -r '.shape[0]' $F/fields.json); ny=$(jq -r '.shape[1]' $F/fields.json); nx=$(jq -r '.shape[2]' $F/fields.json); dir=$(jq -r .dir $REF/model.json)
args=()
for k in $(seq 0 $(( (ze + 63) / 64 - 1 ))); do
  s=$(printf 's%03d' $k); d=$O/slabs/$s; mkdir -p $d
  c0=$((k * 64)); c1=$((c0 + 64)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - 16)); [ $a -lt 0 ] && a=0; b=$((c1 + 16)); [ $b -gt $ze ] && b=$ze
  if [ ! -e $d/done ]; then
    rm -rf $d/f $d/fit; bash /home/forrest/ufsm-fibfit/scripts/wbench/mkcrop.sh $F $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null
    $U fit $d/f --cpu --out $d/fit --dir $dir --ref-model $REF --ref-snap 1 --uw-repair 0 --crop-material 2 $X > $d/fit.log 2>&1; rc=$?
    echo "$s rc $rc $(date +%T)" >> $O/slabs.txt
    if [ $rc = 0 ]; then touch $d/done; elif grep -q "no data for the prior" $d/fit.log; then touch $d/empty $d/done; fi
    rm -rf $d/f
  fi
  [ -e $d/empty ] || args+=($d/fit $((2 * c0)) $((2 * c1)))
done
rm -rf $O/joined; $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}" > $O/join.log 2>&1 || { echo join failed >> $O/slabs.txt; exit 1; }
rm -rf $O/wraps; $U wraps $O/joined $O/wraps --step 4 --out-um 9.596 --sfc 0.25 --fields $F --threads 8 --min-cm2 0.5 > $O/wraps.log 2>&1 || { echo wraps failed >> $O/slabs.txt; exit 1; }
rm -rf $O/wraps_snap; $U wrap-snap $O/wraps $O/wraps_snap --target mix --axis $S/axis.json --axis-um 2.399 --ct /vesuvius/ufsm/v2/ct PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr/2 --ct-um 9.596 --fields $F --threads 8 > $O/snap.log 2>&1 || { echo snap failed >> $O/slabs.txt; exit 1; }
$U wrap-dump $O/wraps_snap $O/ours4.f32 --z -10,1e9 --every 4 --grid-um 2.399 > $O/dump.log 2>&1
echo "pipe done $(date +%T)" >> $O/slabs.txt

#!/usr/bin/env bash
# scripts/fibers/wbfib.sh CKPT OUTDIR [BOXES]: our own fibres for the wbench boxes, for fit --fibers (fibre votes in the slip repair).
# Per box (boxes.tsv; default the core list): predict the H / V fibre mask model (scripts/fibers/train_hv.sh) over the box + MG
# label-level voxels, trace lines through each mask (fiber-trace --mask, level-0 points), and export them into the box-local frame of
# the box's fit fields (fiber-export with UFIB inputs; FIELDS = a wbench fields tag, default lat-wind96-ct: wind96 at window 384, all 44 core boxes) -> OUTDIR/BOX.fib.
# Then: wbench.sh with FIELDS=wind96 FIT="--fibers OUTDIR/{box}.fib --uw-rep-fib 16" against the same fields without FIT.
set -u
CK=${1:?ckpt}; OD=${2:?outdir}; BX=${3:-core}; GPU=${GPU:-0}; MG=${MG:-16}; FT=${FIELDS:-lat-wind96-ct}
U=${U:-/home/forrest/ufsm-fib/build/ufsm}; D=$(dirname "$(readlink -f "$0")")
source $D/../wbench/site-desk.env
BT=$WB/boxes.tsv; mkdir -p $OD
if [ "$BX" = core ]; then BL=$(grep -v '^#' $BT | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1); else BL=$BX; fi
for B in $BL; do
  [ -s $OD/$B.fib ] && { echo "$B done"; continue; }
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"
  case $sc in
    paris4|paris4j) R=$P4ROOT; V=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; UM=2.4; LV=2;;
    1667) R=$MIRROR; V=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr; UM=2.399; LV=2;;
    *) echo "$B: scroll $sc not handled"; continue;;
  esac
  f=$WB/fields/$FT/$B; [ -e $f/fields.json ] || { echo "$B: no fields $f"; continue; }
  w=$OD/work/$B; mkdir -p $w
  bz=$((z - MG)); by=$((y - MG)); bx=$((x - MG)); [ $bz -lt 0 ] && bz=0; [ $by -lt 0 ] && by=0; [ $bx -lt 0 ] && bx=0
  BOX=$bz,$by,$bx,$((nz + 2 * MG)),$((ny + 2 * MG)),$((nx + 2 * MG)); fl=""
  for c in 0 1; do F=$([ $c = 0 ] && echo H || echo V)
    [ -e $w/$F/zarr.json ] || $U predict $CK $R $V $w/$F --um $UM --level $LV --box $BOX --channel $c --gpu $GPU --threads 6 --cache $CACHE > $w/predict_$F.log 2>&1 \
      || { echo "$B predict $F failed: $w/predict_$F.log"; continue 2; }
    nice -n 10 $U fiber-trace - - $w/$F.fib --mask $w/$F $(ls $w/$F | grep -v zarr.json | sort -g | head -1) --mask-origin $bz,$by,$bx --box $BOX \
      --scaledown $LV --family $F --half 1024 --min-len 48 --threads 4 > $w/trace_$F.log 2>&1 || { echo "$B trace $F failed: $w/trace_$F.log"; continue 2; }
    fl=${fl:+$fl,}$w/$F.fib
  done
  $U fiber-export $fl $f $OD/$B.fib --level $LV --margin 8 > $w/export.log 2>&1 || { echo "$B export failed"; continue; }
  echo "$B $(tail -1 $w/export.log)"
done

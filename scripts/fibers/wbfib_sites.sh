#!/usr/bin/env bash
# scripts/fibers/wbfib_sites.sh CKPT OUTDIR [SITES]: our own fibres for the wbench Paris 4 slab sites (wbench.sh SLABS=1), for
# fit-join --fibers OUTDIR/site<s>.fib --join-fib W (wbench.sh JOIN=...). Per site: the joined crop fC of the site's fields
# (FIELDS tag, default wind96; wbench.sh's mkcrop 0 96 96 160 3328 3520), a z band of +- BAND native voxels around the A|B boundary
# (fC-local native z 160; default 72 = 0.7 mm at 9.6 um, the votes reach 0.6 mm) over the whole crop, FAMS (default V: vertical
# fibres run along z and cross the boundary, horizontal ones mostly do not; tracing is the cost, ~15 min per 2 Gvox tile and family), predicted (the H / V fibre mask model) and traced in NT y tiles (the tracer holds three
# bytes per voxel), then exported into fC's box-local frame. Masks are deleted after tracing (a band is ~2 GB per tile and channel).
set -u
CK=${1:?ckpt}; OD=${2:?outdir}; SITES=${3:-4624 5392 5904 6672}; GPU=${GPU:-0}; FT=${FIELDS:-wind96}; BAND=${BAND:-72}; NT=${NT:-4}; FAMS=${FAMS:-V}
U=${U:-/home/forrest/ufsm-fib/build/ufsm}; D=$(dirname "$(readlink -f "$0")")
source $D/../wbench/site-desk.env
R=$P4ROOT; V=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; UM=2.4; LV=2
mkdir -p $OD
for s in $SITES; do
  S=site$s; [ -s $OD/$S.fib ] && { echo "$S done"; continue; }
  w=$OD/work/$S; mkdir -p $w
  bash $D/../wbench/mkcrop.sh $WB/fields/$FT/$S $w/fC 0 96 96 160 3328 3520 > /dev/null || { echo "$S: mkcrop failed"; continue; }
  read -r oz oy ox nz ny nx h <<< "$(jq -r '[.origin[0], .origin[1], .origin[2], .shape[0], .shape[1], .shape[2], .h] | @tsv' $w/fC/fields.json)"
  bz=$((oz + 160 - BAND)); NY=$((ny * h)); NX=$((nx * h)); ty=$(( (NY + NT - 1) / NT )); fl=""
  for t in $(seq 0 $((NT - 1))); do
    y0=$((oy + t * ty)); hy=$ty; [ $((t * ty + hy)) -gt $NY ] && hy=$((NY - t * ty)); BOX=$bz,$y0,$ox,$((2 * BAND)),$hy,$NX
    for F in $FAMS; do c=$([ $F = H ] && echo 0 || echo 1); o=$w/t$t$F
      [ -s $o.fib ] && { fl=${fl:+$fl,}$o.fib; continue; }
      flock /vesuvius/ufsm/wbench/gpu$GPU.lock $U predict $CK $R $V $o --um $UM --level $LV --box $BOX --channel $c --gpu $GPU --threads 6 --cache $CACHE > $o.predict.log 2>&1 \
        || { echo "$S tile $t predict $F failed: $o.predict.log"; continue 3; }
      nice -n 10 ~/bin/memcap -m 16G $U fiber-trace - - $o.fib --mask $o 9.6 --mask-origin $bz,$y0,$ox --box $BOX --scaledown $LV --family $F --half 1024 \
        --min-len 48 --threads 8 > $o.trace.log 2>&1 || { echo "$S tile $t trace $F failed: $o.trace.log"; continue 3; }
      rm -rf $o; fl=${fl:+$fl,}$o.fib
    done
  done
  $U fiber-export $fl $w/fC $OD/$S.fib --level $LV --margin 8 > $w/export.log 2>&1 || { echo "$S export failed"; continue; }
  echo "$S $(tail -1 $w/export.log)"
done

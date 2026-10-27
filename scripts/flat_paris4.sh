#!/usr/bin/env bash
# flat_paris4.sh: SLIM-flatten Paris 4 wraps (ufsm flatten) and render CT / ink96 / gambiarra2 (published-teacher 3D ink) on the
# original and the flattened grids, full (SC px per 9.6 um voxel) and half resolution JPEGs. Desk paths.
# Env: W (wraps dir, .sfc or tifxyz), WR (wrap list), TAG (sub-directory of $F per wrap set), SKIPFLAT=1 (the flat tifxyz already in
# $F/$TAG/flat, e.g. flattened elsewhere), PUB=1 (w072: the old published render $PUBJPG warped onto the flat grid + a side-by-side).
# Outputs $F/$TAG/out/<w>/<w>_{orig,flat}_{full,half}_{ct,ink96,gamb}.jpg, $F/$TAG/flatten.log (distortion statistics).
set -u
F=${F:-/vesuvius/ufsm/flat}; U=${U:-$F/src/build/ufsm}; P=/vesuvius/ufsm/paris4
W=${W:-$P/slabs-orc/wraps}; R=/vesuvius/ufsm/volcomp; V=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr
I96=$P/ink/ink96.zarr; G=/vesuvius/ufsm/gt/paris4-teach-20261026/gambiarra2.zarr
WR=${WR:-w060,w072,w103,w123}; TH=${TH:-16}; SC=${SC:-2}; FLAGS=${FLAGS:-}; TAG=${TAG:-orc}
PUBJPG=${PUBJPG:-$P/slabs-orc/renders/w072/w072_published.jpg}   # (render --scale 4 of the step-4 wrap, halved: 8 px per grid step)
PXUM=$(awk -v s=$SC 'BEGIN { print 9.6 / s }')
D=$F/$TAG; mkdir -p $D/orig $D/flat $D/r $D/out
$U tifxyz-scale $W $D/orig 1 --wraps $WR 2>&1 | tail -1
if [ "${SKIPFLAT:-0}" != 1 ]; then
  /usr/bin/time -v $U flatten $D/orig $D/flat --wraps $WR --threads $TH $FLAGS > $D/flatten.log 2> $D/flatten.err || { echo flatten failed; tail $D/flatten.err; exit 1; }
  cat $D/flatten.log
fi
for n in ${WR//,/ }; do
  O=$D/out/$n; mkdir -p $O
  for k in orig flat; do
    d=$D/r/$k/$n; rm -rf $d; mkdir -p $d/in; ln -sfn $D/$k/$n $d/in/$n
    UFSM_RENDER_MEM=16 $U render $d/in $d/i96 --volume $I96 9.6 --ct $R $V/2 --depth 1 --scale $SC --threads $TH > $d/render.log 2>&1 &&
    UFSM_RENDER_MEM=16 $U render $d/in $d/gb --volume $G 9.6 --depth 1 --scale $SC --threads $TH >> $d/render.log 2>&1 || { echo "$n $k: render failed"; tail -2 $d/render.log; continue; }
    INKS=(--ink "ink96=$d/i96/$n.tif" --ink "gamb=$d/gb/$n.tif")
    $U ink-compare - --ct $d/i96/${n}_ct.tif "${INKS[@]}" --px-um $PXUM --stretch --stretch-ct --full --each $O/${n}_${k}_full_ --each-ct > $O/${n}_${k}_stats.txt 2> /dev/null
    $U ink-compare - --ct $d/i96/${n}_ct.tif "${INKS[@]}" --px-um $PXUM --stretch --stretch-ct --each $O/${n}_${k}_half_ --each-ct > /dev/null 2>&1
    if [ "${PUB:-0}" = 1 ] && [ $n = w072 ] && [ $k = flat ] && [ -f $PUBJPG ]; then   # the published render, warped flat; side by side
      $U flatten-warp $PUBJPG $D/flat/$n $d/pub.tif --src-px 8 --scale $SC 2>&1 | tail -1
      $U ink-compare $O/${n}_flat_side_half.jpg --ct $d/i96/${n}_ct.tif --ink "published=$d/pub.tif" --ink "ink96=$d/i96/$n.tif" --ink "gamb=$d/gb/$n.tif" --px-um $PXUM --stretch --stretch-ct > /dev/null 2>&1
      $U ink-compare - --ct $d/i96/${n}_ct.tif --ink "published=$d/pub.tif" --px-um $PXUM --full --each $O/${n}_flat_full_ > /dev/null 2>&1
    fi
    echo "$n $k: $(tr '\n' ' ' < $O/${n}_${k}_stats.txt)"
  done
done

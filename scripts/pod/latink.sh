#!/bin/bash
# latink.sh O GPU OUT: the unroll's ink stage (INKCMD) with the latent universal ink head (user 11-13: "forget about ift3"), inkB's recipe
# (sched/inkwatch/u1667ink-v3.sh, GATE=1): predict --heads ink,c12,c13 at the run's CT level over its box, window 544 halo 16, on two tile
# grids (origin 0 and 256), each head blended with ufsm ink-blend (sin^2 tile weights, tile 512) -> OUT (ink, key = the level's um) and
# O/ink/c12.zarr, O/ink/c13.zarr (p_recto, p_verso; strip_refill.sh gates the rendered ink by max of the two when INKGATE names them).
# The grid-256 pass writes 256 shards (inkB 11-13: with 512 shards each shard overlapped 8 shifted windows, ~7x the windows of grid 0).
# Env: LATINK_CK (default /root/pod/ckpt/inkpod-9000.ckpt), ITH (threads).
set -u
O=${1:?}; G=${2:-0}; OUT=${3:?}; U=${U:-/root/pod/ufsm/build/ufsm}; CK=${LATINK_CK:-${POD:-/root/pod}/ckpt/inkpod-9000.ckpt}
row=$(tail -1 $O/bin/scrolls.tsv); R=$(echo "$row" | cut -f2); K=$(echo "$row" | cut -f3); UM=$(echo "$row" | cut -f4); LV=$(echo "$row" | cut -f5)
LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }'); BOX=$(cat $O/box.txt); T=$O/ink/lat.tmp; TH=${ITH:-12}
echo "latink: ck $CK, $R $K level $LV ($LUM um), box $BOX"
mkdir -p $T
for g in 0 256; do
  [ -e $T/g$g.done ] && continue; rm -rf $T/g$g
  $U predict $CK $R $K $T/g$g --um $UM --level $LV --box $BOX --full-grid 1 --occupancy $((LV + 3)) --heads ink,c12,c13 --gpu $G --threads $TH --shard $([ $g = 256 ] && echo 256 || echo 512) \
    --window 544 --halo 16 --grid-origin $g,$g,$g || { echo "latink: predict grid $g failed"; exit 1; }
  for h in ink c12 c13; do [ -e $T/g$g/$h/zarr.json ] || { echo "latink: no $h in $T/g$g"; exit 1; }; done
  touch $T/g$g.done
done
for h in ink c12 c13; do
  d=$OUT; [ $h = ink ] || d=$O/ink/$h.zarr; rm -rf $d
  $U ink-blend $d --um $LUM --tile 512,512,512 --grids 0,0,0:256,256,256 $T/g0/$h $T/g256/$h --threads $TH || { echo "latink: ink-blend $h failed"; exit 1; }
  [ -e $d/zarr.json ] || { echo "latink: no blended $h at $d"; exit 1; }
done
rm -rf $T; echo "latink: done -> $OUT (+ c12.zarr c13.zarr)"

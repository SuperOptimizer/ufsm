#!/usr/bin/env bash
# render.sh S: renders of scroll S's surfaces (desk; A/<S>/surf from fit.sh, A/<S>/ink from ink.sh), C only:
#  surf-snap onto the CT ridges (level 0, +-40 um) -> ufsm render-sheets (CT depth 2, ink depths 5,3,1 native voxels; the tifxyz is on
#  the scan's native grid = CT level 0 = the ink zarr's level 0): half-res JPEG CT | CT + red ink | ink +-5 | +-3 | +-1 per surface
#  -> renders/<surface>.jpg; ufsm ink-compare of the kept CT / ink +-3 renders: CT | ink panel renders/<surface>_ink.jpg + ink statistics
#  -> renders/index.tsv (surface, kind, area cm2, on-papyrus share (fit.sh: surf-stats --ct), ink mean / frac50 / p95 / band).
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "render.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; I=$S/ink/${scan%-masked}-ink96.zarr; K=$s/volumes/$scan.zarr; O=$S/renders; T=$S/rtmp; TH=${TH:-12}
[ -s $S/surf/list.tsv ] && [ -d $I ] || { echo "render.sh: $s needs surf/list.tsv and $I"; exit 1; }
IK=$(ls $I | grep -v json | sort -g | head -1); mkdir -p $O $T; [ -s $O/index.tsv ] || printf 'surface\tkind\tarea_cm2\ton_papyrus\tink_mean\tink_frac50\tink_p95\tink_band\tjpeg\n' > $O/index.tsv
while IFS=$'\t' read -r sf kind area onp; do
  n=$(echo $sf | tr / _); grep -q "^$sf	" $O/index.tsv && continue
  g=$T/$n; rm -rf $g; mkdir -p $g/in
  cp -r $S/surf/$sf $g/in/$n
  C $U surf-snap $g/in $g/snap --volume $R $K/0 --vol-um $um --range 40 --threads $TH > $g/snap.log 2>&1 || { rm -rf $g/snap; cp -r $g/in $g/snap; }
  C $U render-sheets $g/snap $O --ct $R $K/0 --ink $I $IK --ct-div 1 --ink-div 1 --ct-depth 2 --ink-depths 5,3,1 --keep-tif $g/k --threads $TH --mem 8 > $g/render.log 2>&1 || { echo "render.sh: $s $sf render failed"; tail -2 $g/render.log; continue; }
  st=$($U ink-compare $O/${n}_ink.jpg --ct $g/k/ct/$n.tif --ink ink96=$g/k/ink3/$n.tif --px-um $um 2> $g/cmp.log)
  read -r mean f50 p95 band <<< "$(echo "$st" | grep '^ink ink96' | sed -E 's/.*mean ([0-9.]+) frac50 ([0-9.]+) p50 [0-9.]+ p95 ([0-9.]+) p99 [0-9.]+ band ([0-9.]+).*/\1 \2 \3 \4/')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $sf $kind $area ${onp:-na} ${mean:-na} ${f50:-na} ${p95:-na} ${band:-na} $n.jpg >> $O/index.tsv
  rm -rf $g
done < $S/surf/list.tsv
rmdir $T 2> /dev/null
mark $s render "$(($(grep -c . $O/index.tsv) - 1)) surfaces"

#!/usr/bin/env bash
# render.sh S: renders of scroll S's new surfaces (desk, C only), for reading letters. Incremental: every row of A/<S>/surf/list.tsv
# (desk box fits) and A/<S>/surf-l/list.tsv (laptop box fits, pushed) not yet in renders/index.tsv:
#  surf-snap onto the CT ridges (level 0, +-40 um) -> ufsm render-sheets (CT depth 2; ink depths 3 and 1 native voxels = +-26..28 and
#  +-8.6..9.4 um around the recto face) -> renders/<name>_<area>cm2.jpg (half-res CT | CT + red ink | ink +-3 | ink +-1);
#  ufsm ink-compare --stretch (p1-p99) of the kept renders: renders/<name>_<area>cm2_ink.jpg (half-res CT | ink +-1 | ink +-3) and
#  renders/<name>_<area>cm2_crop.jpg (full-res 1024 px crop of the most ink-rich window, --best-crop) + ink statistics
#  -> renders/index.tsv (surface, kind, area cm2, on papyrus, ink mean / frac50 / p95 / band, jpeg); the snapped tifxyz -> sheets/<name>
#  and a row in sheets.tsv (sheet, box, area cm2, on papyrus, tifxyz path on the laptop, scan um) for the ink-model agent.
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "render.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; I=$S/ink/${scan%-masked}-ink96.zarr; K=$s/volumes/$scan.zarr; O=$S/renders; T=$S/rtmp; TH=${TH:-12}
LA=/home/forrest/ufsm-data/allscrolls/$s   # the laptop copy (queue.sh laptop pulls renders/, sheets/, sheets.tsv)
[ -d $I ] || { echo "render.sh: $s has no ink $I"; exit 1; }
IK=$(ls $I | grep -E '^[0-9.]+$' | sort -g | head -1); mkdir -p $O $T $S/sheets
[ -s $O/index.tsv ] || printf 'surface\tkind\tarea_cm2\ton_papyrus\tink_mean\tink_frac50\tink_p95\tink_band\tjpeg\n' > $O/index.tsv
[ -s $S/sheets.tsv ] || printf 'sheet\tbox\tarea_cm2\ton_papyrus\ttifxyz\tscan_um\n' > $S/sheets.tsv
for L in surf surf-l; do
  [ -s $S/$L/list.tsv ] || continue
  while IFS=$'\t' read -r sf kind area onp; do
    grep -q "^$L/$sf	" $O/index.tsv && continue
    n=w_${s}_$(echo $sf | tr / _)$([ $L = surf-l ] && echo _1cm); a2=$(printf '%.1f' $area); nn=${n}_${a2}cm2   # (w*: surf-snap reads w* entries)
    g=$T/$n; rm -rf $g; mkdir -p $g/in; cp -r $S/$L/$sf $g/in/$n
    C $U surf-snap $g/in $g/snap --volume $R $K/0 --vol-um $um --range 40 --threads $TH > $g/snap.log 2>&1 || { rm -rf $g/snap; cp -r $g/in $g/snap; }
    C $U render-sheets $g/snap $g/o --ct $R $K/0 --ink $I $IK --ct-div 1 --ink-div 1 --ct-depth 2 --ink-depths 3,1 --keep-tif $g/k --threads $TH --mem 8 > $g/render.log 2>&1 ||
      { echo "render.sh: $s $sf render failed"; tail -2 $g/render.log; rm -rf $g; continue; }
    [ -e $g/o/$n.jpg ] && mv $g/o/$n.jpg $O/$nn.jpg
    st=$($U ink-compare $O/${nn}_ink.jpg --ct $g/k/ct/$n.tif --ink ink+-1=$g/k/ink1/$n.tif --ink ink+-3=$g/k/ink3/$n.tif --px-um $um --stretch 2> $g/cmp.log)
    $U ink-compare $O/${nn}_crop.jpg --ct $g/k/ct/$n.tif --ink ink+-1=$g/k/ink1/$n.tif --ink ink+-3=$g/k/ink3/$n.tif --px-um $um --stretch --full --best-crop 1024 > /dev/null 2>> $g/cmp.log
    read -r mean f50 p95 band <<< "$(echo "$st" | grep '^ink ink+-3' | sed -E 's/.*mean ([0-9.]+) frac50 ([0-9.]+) p50 [0-9.]+ p95 ([0-9.]+) p99 [0-9.]+ band ([0-9.]+).*/\1 \2 \3 \4/')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $L/$sf $kind $area ${onp:-na} ${mean:-na} ${f50:-na} ${p95:-na} ${band:-na} $nn.jpg >> $O/index.tsv
    rm -rf $S/sheets/$nn; mv $g/snap/$n $S/sheets/$nn
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $nn ${sf%%/*} $area ${onp:-na} $LA/sheets/$nn $um >> $S/sheets.tsv
    rm -rf $g
  done < $S/$L/list.tsv
done
rmdir $T 2> /dev/null
mark $s render "$(($(grep -c . $O/index.tsv) - 1)) surfaces"

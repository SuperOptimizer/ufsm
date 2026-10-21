#!/usr/bin/env bash
# render.sh S: renders of scroll S's new surfaces (desk, C only), for reading letters. Incremental: every row of A/<S>/surf/list.tsv
# (desk box fits) and A/<S>/surf-l/list.tsv (laptop box fits, pushed) not yet in renders/index.tsv:
#  surf-snap onto the CT ridges (level 0, +-40 um); both faces of the sheet: the tifxyz moved FACE voxels (4) along -n and +n
#  (tifxyz-scale --normal, +n = column x row direction of the grid) and the ink rendered FACE deep around each (one-sided bands
#  [-2 FACE, 0] and [0, 2 FACE] voxels = ~0..70 um), plus the thin centre band (+-1 voxel) and CT (depth 2). RDIR=renders-x: another
#  output dir (comparisons; no sheets export):
#   renders/<name>_<area>cm2.jpg        half-res CT | CT + red ink | ink +-3 | ink +-1 (render-sheets overlay)
#   renders/<name>_<area>cm2_ink.jpg    half-res CT | neg face | pos face | centre, each ink panel p1-p99 stretched per sheet
#   renders/<name>_<area>cm2_fixed.jpg  the same on ONE fixed ink scale (bytes 0..FIXHI = p 0..0.5) for every sheet: blank sheets stay dark
#   renders/<name>_<area>cm2_crop.jpg   full-res 1024 px crop of the richer face's most ink-rich window, fixed scale
#  -> renders/index.tsv (surface, kind, area, on papyrus, ink mean neg / pos / centre, richer face, its frac50 / p95 / band, jpeg);
#  the snapped tifxyz -> sheets/<name> + a row in sheets.tsv (sheet, box, area cm2, on papyrus, tifxyz path on the laptop, scan um).
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "render.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; I=$S/ink/${scan%-masked}-ink96.zarr; K=$s/volumes/$scan.zarr; O=$S/${RDIR:-renders}; T=$S/rtmp-${RDIR:-renders}; TH=${TH:-12}; FACE=${FACE:-4}; FIXHI=${FIXHI:-128}
LA=/home/forrest/ufsm-data/allscrolls/$s   # the laptop copy (queue.sh laptop pulls renders/, sheets/, sheets.tsv)
[ -d $I ] || { echo "render.sh: $s has no ink $I"; exit 1; }
IK=$(ls $I | grep -E '^[0-9.]+$' | sort -g | head -1); mkdir -p $O $T $S/sheets
[ -s $O/index.tsv ] || printf 'surface\tkind\tarea_cm2\ton_papyrus\tink_neg\tink_pos\tink_centre\tricher\tfrac50\tp95\tband\tjpeg\n' > $O/index.tsv
[ -s $S/sheets.tsv ] || printf 'sheet\tbox\tarea_cm2\ton_papyrus\ttifxyz\tscan_um\n' > $S/sheets.tsv
st1() { echo "$1" | grep "^ink $2:" | sed -E 's/.*mean ([0-9.]+) frac50 ([0-9.]+) p50 [0-9.]+ p95 ([0-9.]+) p99 [0-9.]+ band ([0-9.]+).*/\1 \2 \3 \4/'; }
for L in surf surf-l; do
  [ -s $S/$L/list.tsv ] || continue
  while IFS=$'\t' read -r sf kind area onp; do
    grep -q "^$L/$sf	" $O/index.tsv && continue
    n=w_${s}_$(echo $sf | tr / _)$([ $L = surf-l ] && echo _1cm); a2=$(printf '%.1f' $area); nn=${n}_${a2}cm2   # (w*: surf-snap reads w* entries)
    g=$T/$n; rm -rf $g; mkdir -p $g/in; cp -r $S/$L/$sf $g/in/$n
    C $U surf-snap $g/in $g/snap --volume $R $K/0 --vol-um $um --range 40 --threads $TH > $g/snap.log 2>&1 || { rm -rf $g/snap; cp -r $g/in $g/snap; }
    C $U render-sheets $g/snap $g/o --ct $R $K/0 --ink $I $IK --ct-div 1 --ink-div 1 --ct-depth 2 --ink-depths 3,1 --keep-tif $g/k --threads $TH --mem 8 > $g/render.log 2>&1 ||
      { echo "render.sh: $s $sf render failed"; tail -n 2 $g/render.log; rm -rf $g; continue; }
    [ -e $g/o/$n.jpg ] && mv $g/o/$n.jpg $O/$nn.jpg
    for f in m p; do sg=-; [ $f = p ] && sg=
      C $U tifxyz-scale $g/snap $g/s$f 1 --normal $sg$FACE > $g/face.log 2>&1 &&
      UFSM_RENDER_MEM=8 C $U render $g/s$f $g/r$f --volume $I $IK --depth $FACE --scale 1 --threads $TH >> $g/face.log 2>&1 || echo "render.sh: $s $sf face $f failed"
    done
    INKS=(--ink "neg face=$g/rm/$n.tif" --ink "pos face=$g/rp/$n.tif" --ink "centre=$g/k/ink1/$n.tif")
    st=$($U ink-compare $O/${nn}_ink.jpg --ct $g/k/ct/$n.tif "${INKS[@]}" --px-um $um --stretch --stretch-ct 2> $g/cmp.log)
    $U ink-compare $O/${nn}_fixed.jpg --ct $g/k/ct/$n.tif "${INKS[@]}" --px-um $um --fixed 0,$FIXHI --stretch-ct > /dev/null 2>> $g/cmp.log
    read -r mn _ <<< "$(st1 "$st" 'neg face')"; read -r mp _ <<< "$(st1 "$st" 'pos face')"; read -r mc _ <<< "$(st1 "$st" centre)"
    rich=neg; awk -v a=${mp:-0} -v b=${mn:-0} 'BEGIN { exit !(a > b) }' && rich=pos
    read -r _ f50 p95 band <<< "$(st1 "$st" "$rich face")"
    $U ink-compare $O/${nn}_crop.jpg --ct $g/k/ct/$n.tif --ink "$rich face=$g/r${rich:0:1}/$n.tif" --ink "centre=$g/k/ink1/$n.tif" --px-um $um --fixed 0,$FIXHI --stretch-ct --full --best-crop 1024 > /dev/null 2>> $g/cmp.log
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $L/$sf $kind $area ${onp:-na} ${mn:-na} ${mp:-na} ${mc:-na} $rich ${f50:-na} ${p95:-na} ${band:-na} $nn.jpg >> $O/index.tsv
    [ -n "${RDIR:-}" ] && { rm -rf $g; continue; }
    rm -rf $S/sheets/$nn; mv $g/snap/$n $S/sheets/$nn
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $nn ${sf%%/*} $area ${onp:-na} $LA/sheets/$nn $um >> $S/sheets.tsv
    rm -rf $g
  done < $S/$L/list.tsv
done
rmdir $T 2> /dev/null
[ -n "${RDIR:-}" ] || mark $s render "$(($(grep -c . $O/index.tsv) - 1)) surfaces"

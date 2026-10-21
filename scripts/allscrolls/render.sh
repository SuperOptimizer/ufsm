#!/usr/bin/env bash
# render.sh S: renders of scroll S's new surfaces (desk, C only), for reading letters. Incremental: every row of A/<S>/surf/list.tsv
# (desk box fits, self-gated) and A/<S>/surf-l/list.tsv (laptop box fits, pushed) not yet in renders/index.tsv.
# Per sheet one folder A/<S>/sheets/<sheet>/ (the laptop pulls it to ~/ufsm-data/allscrolls/<S>/sheets/<sheet>/): the surf-snapped
# tifxyz (CT ridges, level 0, +-40 um) and every panel as its own half-res JPEG on one pixel grid, no labels (user 10-20):
#   <sheet>_ct.jpg                          CT (+-1 voxel), p1-p99 stretched
#   <sheet>_3dink.jpg                       ink, centre band (+-1 voxel)
#   <sheet>_3dink_in.jpg / _3dink_out.jpg   ink on the INWARD face (toward the scroll axis; the published segments' / ink-ext --inward
#                                           convention) and the outward one: the tifxyz moved FACE (4) voxels along the normal oriented
#                                           toward the axis at the sheet (tifxyz-scale --normal --toward; our fits' raw +n points outward),
#                                           max over +-FACE around it (~0..70 um one-sided). These three p1-p99 stretched per sheet.
#   <sheet>_3dink{,_in,_out}_fixed.jpg      the same on ONE fixed ink scale (bytes 0..FIXHI = p 0..0.5) for every sheet
#   <sheet>_3dink_crop.jpg                  full-res 1024 px crop of the richer face's most ink-rich window, fixed scale
# (the ink-model agent adds <sheet>_reader3 / _hecate96 / _v8in / _ensemble.jpg to the folder).
#  -> renders/index.tsv (surface, kind, kept area, on papyrus, masked share, ink mean in / out / centre, richer face, its frac50 /
#  p95 / band, sheet folder on the laptop) and sheets.tsv (sheet, box, area cm2, on papyrus, tifxyz path on the laptop, scan um).
#  RDIR=x: the index in A/<S>/x, the sheet folders in A/<S>/x/sheets (comparisons, e.g. FACE=6 RDIR=renders-face6; no sheets.tsv).
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "render.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; I=$S/ink/${scan%-masked}-ink96.zarr; K=$s/volumes/$scan.zarr; O=$S/${RDIR:-renders}; T=$S/rtmp-${RDIR:-renders}; TH=${TH:-12}; FACE=${FACE:-4}; FIXHI=${FIXHI:-128}
SH=$S/sheets; LA=/home/forrest/ufsm-data/allscrolls/$s/sheets; [ -n "${RDIR:-}" ] && { SH=$O/sheets; LA=/home/forrest/ufsm-data/allscrolls/$s/$RDIR/sheets; }
[ -d $I ] || { echo "render.sh: $s has no ink $I"; exit 1; }
AX=$S/axis.tsv   # the scroll axis per native z (y x), from the published fit fields' fields.json
[ -s $AX ] || curl -sf --retry 5 $R/$s/representations/predictions/fit-fields/$fz/fields.json | jq -r '.axis[] | @tsv' > $AX
IK=$(ls $I | grep -E '^[0-9.]+$' | sort -g | head -1); mkdir -p $O $T $SH
[ -s $O/index.tsv ] || printf 'surface\tkind\tarea_cm2\ton_papyrus\tmasked\tink_in\tink_out\tink_centre\tricher\tfrac50\tp95\tband\tsheet_dir\n' > $O/index.tsv
[ -n "${RDIR:-}" ] || [ -s $S/sheets.tsv ] || printf 'sheet\tbox\tarea_cm2\ton_papyrus\ttifxyz\tscan_um\n' > $S/sheets.tsv
st1() { echo "$1" | grep "^ink $2:" | sed -E 's/.*mean ([0-9.]+) frac50 ([0-9.]+) p50 [0-9.]+ p95 ([0-9.]+) p99 [0-9.]+ band ([0-9.]+).*/\1 \2 \3 \4/'; }
for L in surf surf-l; do
  [ -s $S/$L/list.tsv ] || continue
  while IFS=$'\t' read -r sf kind area onp masked; do
    grep -q "^$L/$sf	" $O/index.tsv && continue
    n=w_${s}_$(echo $sf | tr / _)$([ $L = surf-l ] && echo _1cm); a2=$(printf '%.1f' $area); nn=${n}_${a2}cm2   # (w*: the C tools read w* entries)
    g=$T/$n; rm -rf $g; mkdir -p $g/in; cp -r $S/$L/$sf $g/in/$n
    C $U surf-snap $g/in $g/snap --volume $R $K/0 --vol-um $um --range 40 --threads $TH > $g/snap.log 2>&1 || { rm -rf $g/snap; cp -r $g/in $g/snap; }
    UFSM_RENDER_MEM=8 C $U render $g/snap $g/rc --volume $I $IK --ct $R $K/0 --depth 1 --scale 1 --threads $TH > $g/render.log 2>&1 ||
      { echo "render.sh: $s $sf render failed"; tail -n 2 $g/render.log; rm -rf $g; continue; }
    zc=$(jq -r '(.bbox[0][2] + .bbox[1][2]) / 2 | floor' $g/snap/$n/meta.json); read -r ay ax <<< "$(sed -n "$((zc + 1))p" $AX)"   # the axis at the sheet
    for f in in out; do sg=; [ $f = out ] && sg=-   # +FACE toward the axis = the inward face, -FACE the outward one
      C $U tifxyz-scale $g/snap $g/s$f 1 --normal $sg$FACE --toward $ay,$ax > $g/face-$f.log 2>&1 &&
      UFSM_RENDER_MEM=8 C $U render $g/s$f $g/r$f --volume $I $IK --depth $FACE --scale 1 --threads $TH >> $g/face-$f.log 2>&1 || echo "render.sh: $s $sf face $f failed"
    done
    d=$SH/$nn; rm -rf $d; mkdir -p $d; for f in x.tif y.tif z.tif meta.json; do cp $g/snap/$n/$f $d/; done
    INKS=(--ink "3dink=$g/rc/$n.tif" --ink "3dink_in=$g/rin/$n.tif" --ink "3dink_out=$g/rout/$n.tif")
    st=$($U ink-compare - --ct $g/rc/${n}_ct.tif "${INKS[@]}" --px-um $um --stretch --stretch-ct --each $d/${nn}_ --each-ct 2> $g/cmp.log)
    $U ink-compare - --ct $g/rc/${n}_ct.tif "${INKS[@]}" --px-um $um --fixed 0,$FIXHI --each $d/${nn}_ --each-suffix _fixed > /dev/null 2>> $g/cmp.log
    read -r mi _ <<< "$(st1 "$st" 3dink_in)"; read -r mo _ <<< "$(st1 "$st" 3dink_out)"; read -r mc _ <<< "$(st1 "$st" 3dink)"
    rich=in; awk -v a=${mo:-0} -v b=${mi:-0} 'BEGIN { exit !(a > b) }' && rich=out
    read -r _ f50 p95 band <<< "$(st1 "$st" 3dink_$rich)"
    $U ink-compare - --ct $g/rc/${n}_ct.tif --ink "3dink=$g/r$rich/$n.tif" --px-um $um --fixed 0,$FIXHI --full --best-crop 1024 \
      --each $d/${nn}_ --each-suffix _crop > /dev/null 2>> $g/cmp.log
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $L/$sf $kind $area ${onp:-na} ${masked:-na} ${mi:-na} ${mo:-na} ${mc:-na} $rich ${f50:-na} ${p95:-na} ${band:-na} $LA/$nn >> $O/index.tsv
    [ -n "${RDIR:-}" ] || printf '%s\t%s\t%s\t%s\t%s\t%s\n' $nn ${sf%%/*} $area ${onp:-na} $LA/$nn $um >> $S/sheets.tsv
    rm -rf $g
  done < $S/$L/list.tsv
done
rmdir $T 2> /dev/null
[ -n "${RDIR:-}" ] || mark $s render "$(($(grep -c . $O/index.tsv) - 1)) surfaces"

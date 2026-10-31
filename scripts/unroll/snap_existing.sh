#!/usr/bin/env bash
# snap_existing.sh SCROLL [OUT]: ufsm wrap-snap over an already unrolled scroll's wraps (the pulled pod / desk runs), then the
# on-papyrus share before / after (surf-stats --ct-only). The snapped wraps go to OUT/SCROLL/wraps_snap (OUT default /mnt/scratch/snap/existing);
# re-run the strip stages on them (unroll.sh's strip / stripflat / stripimg, or scripts/strip/strip_run.sh OUT WRAPS) to get snapped strips.
# Inputs, found in order: RUN (env) or /vesuvius/ufsm/pods/S/run, /mnt/scratch/pods/S/run (wraps/wNNN.sfc, axis.json, fields/ when kept).
# CT: scripts/corpus/list.tsv (root, key, um, level) when the root is on this host, else the mirror over HTTP (MIRROR) with a decoded-chunk
# cache in OUT/S/ctcache (deleted at the end, KEEPCACHE=1 keeps it). Target: SNAPT (default mix when the run kept its fields' pc plane, else ct).
# Env: U (ufsm), CTROOT (local CT root holding the list key, overrides list.tsv's), TH (threads, 16), SNAPX (extra wrap-snap flags), MEASURE (1: surf-stats before / after).
# Prints: wall s, max RSS GB, CT cache GB, on-papyrus before / after. Never touches the source run.
set -u
S=${1:?usage: snap_existing.sh SCROLL [OUT]}; OUT=${2:-/mnt/scratch/snap/existing}
H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $H/../.. && pwd)
U=${U:-$REPO/build/ufsm}; TH=${TH:-16}; MIRROR=${MIRROR:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
RUN=${RUN:-}; for r in /vesuvius/ufsm/pods/$S/run /mnt/scratch/pods/$S/run; do [ -z "$RUN" ] && [ -d $r/wraps ] && [ -e $r/axis.json ] && RUN=$r; done
[ -n "$RUN" ] || { echo "snap_existing: no run with wraps/ + axis.json for $S (set RUN)"; exit 2; }
row=$(awk -F'\t' -v s=$S '$1 == s' $REPO/scripts/corpus/list.tsv $H/scrolls.tsv | head -1)
[ -n "$row" ] || { echo "snap_existing: $S not in scripts/corpus/list.tsv or scrolls.tsv"; exit 2; }
IFS=$'\t' read -r _ R K UM LV _ <<< "$row"
[ -n "${CTROOT:-}" ] && R=$CTROOT   # (a local copy of the scan elsewhere, e.g. the v2 CT tree /vesuvius/ufsm/v2/ct)
LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }')
O=$OUT/$S; mkdir -p $O; C=
[ -e $R/$K/$LV/zarr.json ] || { R=$MIRROR; K=$(dirname $K)/volumes/$(basename $K); C=$O/ctcache; mkdir -p $C; }   # (the mirror: S/volumes/<scan>.zarr)
T=${SNAPT:-}; [ -n "$T" ] || { [ -e $RUN/fields/pc/zarr.json ] && T=mix || T=ct; }
FL=; [ "$T" != ct ] && FL="--fields $RUN/fields"
echo "$(date '+%F %T') snap_existing $S: run $RUN, CT $R $K/$LV ($LUM um), target $T, out $O" | tee $O/snap.log
rm -rf $O/wraps_snap
/usr/bin/time -v nice -n 5 $U wrap-snap $RUN/wraps $O/wraps_snap --target $T --axis $RUN/axis.json --axis-um $UM --ct $R $K/$LV --ct-um $LUM $FL \
  --threads $TH ${C:+--cache $C} ${SNAPX:-} >> $O/snap.log 2>&1 || { echo "snap_existing: wrap-snap failed (see $O/snap.log)"; exit 1; }
wall=$(awk -F': ' '/Elapsed \(wall/ { n = split($2, a, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + a[i]; print s }' $O/snap.log)
rss=$(awk '/Maximum resident/ { printf "%.1f", $NF / 1048576 }' $O/snap.log); cgb=0; [ -n "$C" ] && cgb=$(du -sk $C | awk '{ printf "%.1f", $1 / 1048576 }')
pap() { [ -d $RUN/fields ] || { echo "-"; return; }; nice -n 5 $U surf-stats $1 --fields $RUN/fields --ct-only 1 --ct $R $K/$LV --ct-um $LUM ${C:+--cache $C} --threads $TH 2>/dev/null |
        sed -n 's/.*on papyrus ([^)]*) \([0-9.]*\);.*/\1/p'; }
b=-; a=-; [ "${MEASURE:-1}" = 1 ] && { b=$(pap $RUN/wraps); a=$(pap $O/wraps_snap); }
echo "$(date '+%F %T') snap_existing $S: $(grep '^wrap-snap: solve' $O/snap.log | tail -1)" | tee -a $O/snap.log
echo "snap_existing $S: wall ${wall} s, max RSS ${rss} GB, CT cache ${cgb} GB, on-papyrus ${b} -> ${a}" | tee -a $O/snap.log
[ -n "$C" ] && [ "${KEEPCACHE:-0}" != 1 ] && rm -rf $C
exit 0

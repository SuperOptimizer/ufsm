#!/usr/bin/env bash
# surflabels.sh S [extra surf-labels flags]: v2 winding labels of scroll S from its wraps (ufsm surf-labels) + the outward-sign check.
# Reads S's row of $TSV (scrolls.tsv: scroll split nwraps wraps wrapcheck strip_json strip_log axis ct_root ct_key ct_um ct_level ...):
#   wraps: $SNAP/S/wraps_snap when present (snapped), else the row's wraps (noted in the log);
#   trim: $TRIM_S env "first,last" if set, else strip.json "wraps" [a, b] -> --first min --last max, else none;
#   grid: the CT level ct_level (local $CTL/ct_key/level) -> --like, --ct (CT 0 = ignore 3), grid um = ct_um x 2^level, axis um = ct_um;
#   gates: wrapcheck dup/merged, ONPAP (default 0.2: --min-onpap with the local CT; 0 = off), PEAKS (e.g. 0.85,1.15; default off);
#   holdouts: $HO boxes of this CT + 0.5 mm -> ignore 4;
#   fixed surfaces: WSRC=<dir> uses <dir>/S/wraps_snap (else <dir>/S/wraps) first (e.g. /vesuvius/ufsm/v2/count/unroll), with its
#   wrapcheck.tsv; GEOM=<dir> adds --geom <dir>/S (geom-check output of those wraps); ROWS=a,b adds --rows a,b (end caps).
# Scope (main 11-02): labels whose turn counts come from our fit alone carry its count errors (P4 4 mm pairs ~.30): they are LOCAL
#   (sheet position, normals, short-range pairs, ink; no long-range winding / count loss). ANCH=<anchor file(s), comma list> (ufsm
#   wrap-anchor from published GT, all GT votes: no --gt-split) adds --anchor: only anchored cells labelled (unanchored -> ignore 7),
#   counts corrected -> GLOBAL. Write the local set to $OUT (default V2/labels) and the anchored set to OUT=V2/labels_anch.
#   $O/scope holds "local" or "global"; scripts/v2/mksources.sh tags each source with it ("scope": "local" | "global").
# Output $OUT/S/{codes,phase,ignore}.zarr/<grid um>, surflabels.log, wraps.tsv, phasecheck.txt, done marker. Env: U (ufsm binary), TH.
set -uo pipefail
S=$1; shift
U=${U:-/mnt/scratch/v2/v2label/src/build/ufsm}; TSV=${TSV:-/vesuvius/ufsm/v2/scrolls.tsv}; SNAP=${SNAP:-/vesuvius/ufsm/v2/wraps_snap}
CTL=${CTL:-/vesuvius/ufsm/v2/ct}; OUT=${OUT:-/vesuvius/ufsm/v2/labels}; HO=${HO:-/vesuvius/ufsm/wbench/holdouts.json}; TH=${TH:-12}
row=$(awk -F'\t' -v s=$S '$1 == s' $TSV); [ -n "$row" ] || { echo "surflabels: $S not in $TSV" >&2; exit 2; }
IFS=$'\t' read -r _ split nw wraps wc sj sl ax ctr ctk ctum ctlev _ <<< "$row"
O=$OUT/$S; mkdir -p $O; rm -f $O/done
W=$wraps; src=unsnapped
[ -d $SNAP/$S/wraps_snap ] && [ -n "$(ls $SNAP/$S/wraps_snap | grep -m1 '\.sfc$')" ] && { W=$SNAP/$S/wraps_snap; src=snapped; }
if [ -n "${WSRC:-}" ]; then for d in wraps_snap wraps; do
  [ -d $WSRC/$S/$d ] && [ -n "$(ls $WSRC/$S/$d | grep -m1 '\.sfc$')" ] && { W=$WSRC/$S/$d; src="fixed $d"; [ -s $WSRC/$S/wrapcheck.tsv ] && wc=$WSRC/$S/wrapcheck.tsv; [ -s $WSRC/$S/axis.json ] && ax=$WSRC/$S/axis.json; break; }; done; fi
trim=()
if [ -n "${TRIM_S:-}" ]; then trim=(--first ${TRIM_S%,*} --last ${TRIM_S#*,})   # (env first: fixed surfaces have their own trim)
elif [ -s "$sj" ]; then read -r a b <<< "$(jq -r '.wraps | "\(min) \(max)"' $sj)"; trim=(--first $a --last $b); fi
[ -n "${GRIDUP:-}" ] && ctlev=$((ctlev + GRIDUP))   # GRIDUP=1: labels on the next coarser CT level (8x smaller / faster; the sampler interpolates a stored phase one level coarser)
ct=$CTL/$ctk/$ctlev; [ -e $ct/zarr.json ] || { echo "surflabels: no local CT $ct" >&2; exit 2; }
gum=$(awk -v u=$ctum -v l=$ctlev 'BEGIN {printf "%.10g", u * 2 ^ l}')
gates=(); [ -s "$wc" ] && gates+=(--wrapcheck $wc); [ -n "${PEAKS:-}" ] && gates+=(--band-peaks $PEAKS)
op=${ONPAP:-0.2}; [ "$op" != 0 ] && gates+=(--min-onpap $op)
[ -n "${GEOM:-}" ] && [ -s ${GEOM}/$S/cells.tsv ] && gates+=(--geom $GEOM/$S); [ -n "${ROWS:-}" ] && gates+=(--rows $ROWS)
an=(); scope=local; [ -n "${ANCH:-}" ] && { an=(--anchor $ANCH); scope=global; }
ho=(); [ -s "$HO" ] && ho=(--exclude $HO --exclude-ct "$(basename $ctk)" --exclude-mm 0.5)
echo "surflabels $S: $src wraps $W, trim ${trim[*]:-none}, grid $gum um (level $ctlev), gates ${gates[*]}, scope $scope ${an[*]}" > $O/surflabels.log
rm -rf $O/codes.zarr $O/phase.zarr $O/ignore.zarr
/usr/bin/time -v nice -n 10 $U surf-labels $W $O --axis $ax --axis-um $ctum --dir auto --grid-um $gum --like $CTL $ctk/$ctlev --ct $CTL $ctk/$ctlev \
  "${trim[@]}" "${gates[@]}" "${an[@]}" "${ho[@]}" --tsv $O/wraps.tsv --threads $TH "$@" >> $O/surflabels.log 2>&1 || { echo "surflabels $S: FAILED" >> $O/surflabels.log; exit 1; }
key=$(ls $O/phase.zarr | grep -v zarr.json | head -1)
$U phase-check $O/phase.zarr $key --axis $ax --scan-um $ctum --phase-um $gum --threads $TH > $O/phasecheck.txt 2>> $O/surflabels.log
echo "$S $src $(grep -o 'kept share [0-9.]*' $O/surflabels.log) $(grep -oE 'outward [0-9.]+' $O/phasecheck.txt) $(grep -oE 'pass|FAIL' $O/phasecheck.txt | tail -1)" | tee -a $O/surflabels.log
echo $scope > $O/scope
touch $O/done

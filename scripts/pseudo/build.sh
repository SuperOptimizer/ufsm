#!/usr/bin/env bash
# scripts/pseudo/build.sh RUN OUT: gated pseudo winding labels of one unroll (DESIGN 2026-11-17 "Fit-derived pseudo labels"), built where
# the unroll ran (on the pod: nothing goes desk -> pod; the labels never leave the machine that built them). Output OUT/:
#   labels/codes.zarr/<gum>   the kept wraps' winding codes (surf-labels; the sampler's phase "onfly" pair rule, rmax 20, reads them)
#   weight.zarr/<gum>         the label weight (phase-coh --coh 0): 0 outside every passing tile, in gt-gate / direction-masked cells,
#                             in air, holdout boxes + EXMM; 1 elsewhere
#   gate.boxes, dir.boxes, agree.boxes   the box-scale inputs (first box wins: agree, dir, then the tile gate)
#   source.json               one sources entry (scope "w4", heads phase / normal / density / wind4 / rv; weight 1: mksources.sh scales)
#   summary.txt, logs/
# Steps (each skipped when its output exists; rm the output to redo):
#   1. unroll-qa on RUN/fields/$FNAME + RUN/joined, every material tile (--tiles 100000), --gate-boxes (conf_lo > 0.4, pw48 > 0.1,
#      w4 < 0.8, fib_pers_h > 130 with FIBRES): s 0 / 1 per tile. The RUN gate: the material-weighted pass share >= RUNMIN (0.5), else
#      REFUSED (exit 3) before any label is built (a P4-like run: passing tiles there still average ids 0.62, geo-gate rows).
#   2. surf-labels on RUN/wraps_snap: RUN/wrapcheck.tsv (dup / merged dropped), state/strip.args (--first / --last), --min-onpap 0.2 with
#      the CT, holdouts HO boxes of the CT + EXMM mm -> ignore, dir from the manifest (refused when missing).
#   3. gt-gate against the scan's m7 surface prob (sheet orientation, ORIENT 25 deg per 64^3 cell, MINKEEP 0.3: good.sh's settings) ->
#      the gated phase (bad cells unknown); skipped with M7=none (then only the tile / direction gates act).
#   4. phase-check --cells on the gated phase -> dir.boxes: 128^3 chunks the aligned test reads reversed (>= 20000 aligned pairs, < 0.5
#      rising outward: scripts/univ/b3_1299_mask.py rule (b)).
#   5. AGREE=CODES_ROOT (another fit's codes.zarr of the same scroll, same grid): surf-labels --gt-cells -> agree.boxes, cells whose
#      pair accuracy (>= 50 pairs) is < AGMIN (0.8) or kept-voxel agreement < AGMIN: 0.
#   6. phase-coh --coh 0 --box-scale (agree + dir + gate) --box-default 0 --papyrus 64 --pap-dilate 2 on the gated phase -> weight.zarr.
#   7. source.json; the full phase rasters are deleted (KEEP=1 keeps them).
# Env: QA (a pulled dense boxes.tsv instead of step 1's unroll-qa: no fields needed), CTROOT (CT root instead of the manifest's), U (ufsm), TH (8), HO (holdouts json, default the 11-14 per-scroll set), EXMM (0.5), RUNMIN (0.5), FNAME (lat), FIBRES (UFIB),
#      M7 (prob ROOT/KEY at the label grid; default: found on the volcomp mirror as good.sh does; "none" skips), CACHE, DRY=1 (print only).
set -uo pipefail
RUN=${1:?RUN dir}; OUT=${2:?OUT dir}
H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}; TH=${TH:-8}; HO=${HO:-/vesuvius/ufsm/univ/scratch1/holdouts-20261114.json}
EXMM=${EXMM:-0.5}; RUNMIN=${RUNMIN:-0.5}; FNAME=${FNAME:-lat}; MIR=https://dl.ash2txt.org/community-uploads/forrest/volcomp
DRY=${DRY:-0}; mkdir -p "$OUT/logs"; L=$OUT/logs/build.log
log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
X() { local lf=$1; shift; if [ "$DRY" = 1 ]; then echo "    $* > $lf"; return 0; fi; "$@" > "$lf" 2>&1; }
M=$RUN/state/manifest.json; [ -s "$M" ] || { log "no $M"; exit 2; }
S=$(basename "$RUN"); S=${S%%-*}
read -r CTR CTK LV GUM <<< "$(jq -r '.ct' "$M" | sed -n 's/^\(\S*\) \(\S*\) level \([0-9]*\) (\([0-9.]*\) um)$/\1 \2 \3 \4/p')"
[ -n "${GUM:-}" ] || { log "cannot parse the manifest ct line: $(jq -r '.ct' "$M")"; exit 2; }
DIR=$(jq -r '.dir // empty' "$M"); case "$DIR" in 1|-1|+1) ;; *) log "REFUSED: the manifest has no spiral dir (got '$DIR')"; exit 2;; esac
CTUM=$(awk -v g=$GUM -v l=$LV 'BEGIN {printf "%.10g", g / 2 ^ l}')   # (the scan's level-0 um: the axis frame)
# CTROOT: build away from the unroll (e.g. on the desk from a pod run pulled to /vesuvius/ufsm/podun/S, lead 11-17): the CT root to read
# instead of the manifest's (the pod's /root/pod/vol); a mirror-style key SCROLL/volumes/SCAN becomes SCROLL/SCAN under a local root
if [ -n "${CTROOT:-}" ]; then CTR=$CTROOT; case "$CTROOT" in *://*) ;; *) CTK=${CTK/\/volumes\//\/};; esac; fi
CTB=$(basename "$CTK"); CK=$(jq -r '.ck_sha // .ck // "?"' "$M")
log "pseudo build $S: run $RUN (ck $CK, dir $DIR), CT $CTR $CTK/$LV at $GUM um, out $OUT"
# 1. the tile gate (and the run gate)
GB=$OUT/gate.boxes
if [ ! -s "$GB" ] && [ -n "${QA:-}" ]; then   # QA: a dense unroll-qa boxes.tsv made where the fields are (pod check stage, UQTILES=100000)
  [ -s "$QA" ] || { log "no QA tiles $QA"; exit 2; }
  python3 "$H/scripts/pseudo/tilegate.py" "$QA" "$GB" > "$OUT/logs/unrollqa.log" || { log "tilegate failed"; exit 1; }
  nt=$(awk 'END {print NR - 1}' "$QA"); [ "$nt" -ge 100 ] || log "WARNING: QA has only $nt tiles (a 48-tile sample, not the dense map?): the weight covers only those tiles"
fi
if [ ! -s "$GB" ]; then
  if [ "$DRY" != 1 ] && { [ ! -d "$RUN/fields/$FNAME" ] || [ ! -d "$RUN/joined" ]; }; then log "no $RUN/fields/$FNAME or $RUN/joined (unroll-qa needs both; kept on the pod until this build ran?)"; exit 2; fi
  fib=(); [ -n "${FIBRES:-}" ] && fib=(--fibres "$FIBRES")
  mkdir -p "$OUT/uq"
  X "$OUT/logs/unrollqa.log" "$U" unroll-qa --fields "$RUN/fields/$FNAME" --model "$RUN/joined" --tiles 100000 --threads "$TH" --out "$OUT/uq" --gate-boxes "$GB.tmp" "${fib[@]}" \
    || { log "unroll-qa failed ($OUT/logs/unrollqa.log)"; exit 1; }
  [ "$DRY" = 1 ] || mv "$GB.tmp" "$GB"
fi
if [ "$DRY" != 1 ]; then
  pass=$(grep -o 'material-weighted [0-9.]*' "$OUT/logs/unrollqa.log" | tail -1 | awk '{print $2}')
  [ -n "$pass" ] || pass=$(awk '!/^#/ {n++; p += $7} END {printf "%.4f", n ? p / n : 0}' "$GB")
  log "tile gate: $(grep -o 'gate boxes .*' "$OUT/logs/unrollqa.log" | tail -1) (pass share used $pass, RUNMIN $RUNMIN)"
  awk -v p="$pass" -v m="$RUNMIN" 'BEGIN {exit !(p >= m)}' || { log "REFUSED: the run gate (material-weighted tile pass share $pass < $RUNMIN): no labels from this run"; echo "$S REFUSED pass $pass" > "$OUT/summary.txt"; exit 3; }
fi
# 2. the labels
LB=$OUT/labels
trim=(); [ -s "$RUN/state/strip.args" ] && read -r -a trim < "$RUN/state/strip.args"
wc=(); [ -s "$RUN/wrapcheck.tsv" ] && wc=(--wrapcheck "$RUN/wrapcheck.tsv")
ho=(); [ -s "$HO" ] && ho=(--exclude "$HO" --exclude-ct "$CTB" --exclude-mm "$EXMM")
if [ ! -e "$LB/done" ]; then
  X "$OUT/logs/surflabels.log" "$U" surf-labels "$RUN/wraps_snap" "$LB" --axis "$RUN/axis.json" --axis-um "$CTUM" --dir "$DIR" --grid-um "$GUM" \
     --like "$CTR" "$CTK/$LV" --ct "$CTR" "$CTK/$LV" --min-onpap 0.2 "${wc[@]}" "${trim[@]}" "${ho[@]}" --threads "$TH" \
    || { log "surf-labels failed ($OUT/logs/surflabels.log)"; exit 1; }
  [ "$DRY" = 1 ] || { touch "$LB/done"; log "labels: $(grep -o 'kept [0-9]* ([0-9.]* mm3), kept share [0-9.]*' "$OUT/logs/surflabels.log")"; }
fi
# 3. gt-gate against the m7 prob
PH=$LB/phase.zarr/$GUM; GP=$OUT/gated
if [ "${M7:-}" != none ] && [ ! -e "$GP/done" ]; then
  if [ -z "${M7:-}" ]; then
    scan=$(echo "$CTB" | cut -d- -f1); PR=$MIR/$S/representations/predictions/surfaces
    pz=$(curl -s "$PR/" | grep -o "href=\"$scan-surface-[^\"]*-m7-L[0-9]-prob.zarr/\"" | head -1 | sed 's/href="//; s|/"||')
    pk=$([ -n "$pz" ] && "$U" info "$PR" "$pz" 2>/dev/null | awk -v g=$GUM 'NR > 1 && $2 > 0 && ($2 / g > 0.99 && $2 / g < 1.01) {print $1; exit}')
    [ -n "$pz" ] && [ -n "$pk" ] && M7="$PR $pz/$pk"
  fi
  if [ -z "${M7:-}" ]; then log "no m7 prob at $GUM um for $S: gt-gate skipped (tile / direction gates only)"; M7=none
  else
    read -r m7r m7k <<< "$M7"; AXS=$(awk -v l=$LV 'BEGIN {printf "%.10g", 2 ^ -l}')
    X "$OUT/logs/gtgate.log" "$U" gt-gate --gt "$LB" --gt-key "codes.zarr/$GUM" --prob "$m7r" --prob-key "$m7k" ${CACHE:+--cache "$CACHE"} --axis "$RUN/axis.json" --axis-scale "$AXS" \
       --out "$GP/gate" --min-support 0 --orient-deg "${ORIENT:-25}" --phase "$LB" --phase-key "phase.zarr/$GUM" --ignore "$LB" --ignore-key "ignore.zarr/$GUM" \
       --phase-out "$GP/phase.zarr" --min-keep "${MINKEEP:-0.3}" --jobs "$TH" || { log "gt-gate failed ($OUT/logs/gtgate.log)"; exit 1; }
    [ "$DRY" = 1 ] || { touch "$GP/done"; log "gt-gate: $(grep -o 'holding [0-9.]* of the sheet voxels' "$OUT/logs/gtgate.log")"; }
  fi
fi
[ "${M7:-}" != none ] && [ -d "$GP/phase.zarr" -o "$DRY" = 1 ] && PR_=$GP/phase.zarr || PR_=$LB/phase.zarr
# 4. the direction mask
DB=$OUT/dir.boxes
if [ ! -s "$DB" ]; then
  X "$OUT/logs/phasecheck.log" "$U" phase-check "$PR_" "$GUM" --axis "$RUN/axis.json" --scan-um "$CTUM" --phase-um "$GUM" --count 1 --boxes 100000000 --cells "$OUT/cells.tsv" --threads "$TH"
  [ "$DRY" = 1 ] || { python3 "$H/scripts/pseudo/dirmask.py" "$OUT/cells.tsv" "$DB" | tee -a "$L"; }
fi
# 5. the two-fit agreement (optional)
AB=$OUT/agree.boxes
if [ -n "${AGREE:-}" ] && [ ! -s "$AB" ]; then
  X "$OUT/logs/agree.log" "$U" surf-labels "$RUN/wraps_snap" "$OUT/agree-tmp" --axis "$RUN/axis.json" --axis-um "$CTUM" --dir "$DIR" --grid-um "$GUM" --like "$CTR" "$CTK/$LV" \
     "${wc[@]}" "${trim[@]}" --gt "$AGREE" "$GUM" --gt-cells "$OUT/agree.tsv" --threads "$TH" || { log "agreement run failed"; exit 1; }
  [ "$DRY" = 1 ] || { python3 "$H/scripts/pseudo/agreemask.py" "$OUT/agree.tsv" "$AB" "${AGMIN:-0.8}" | tee -a "$L"; [ "${KEEP:-0}" = 1 ] || rm -rf -- "$OUT/agree-tmp"; }
fi
# 6. the weight
if [ ! -e "$OUT/weight.zarr/done" ]; then
  [ "$DRY" = 1 ] || cat $( [ -s "$AB" ] && echo "$AB" ) "$DB" "$GB" > "$OUT/all.boxes"
  X "$OUT/logs/weight.log" "$U" phase-coh "$PR_" "$GUM" "$CTR" "$CTK/$LV" "$OUT" --weight 0.3,0.6 --coh 0 --box-scale "$OUT/all.boxes" --box-default 0 --papyrus 64 --pap-dilate 2 --threads "$TH" \
    || { log "phase-coh failed ($OUT/logs/weight.log)"; exit 1; }
  [ "$DRY" = 1 ] || { touch "$OUT/weight.zarr/done"; log "weight: $(grep -E 'box-scale|mean weight' "$OUT/logs/weight.log" | tr '\n' ' ')"; }
fi
# 7. the sources entry
lo=$LV; ax=$RUN/axis.json
jq -n --arg n "$S-pseudo" --arg root "$CTR" --arg ct "$CTK" --argjson um "$CTUM" --argjson lv "$lo" --arg ax "$ax" --arg cz "$LB/codes.zarr" --arg k "$GUM" \
      --arg w "$OUT/weight.zarr" --arg ho "$HO" --arg note "fit-derived pseudo labels (scripts/pseudo/build.sh): run $RUN ck $CK dir $DIR; tile / gt-gate / direction gated weight" \
  '{name: $n, scope: "w4", root: $root, ct: $ct, um: $um, min_level: $lv, max_level: $lv, axis: $ax, iso: 64, weight: 1,
    heads: {phase: 1, normal: 1, density: 1, wind4: 1, rv: 1},
    targets: {band: {radius: 20, span: 19, phase: {onfly: {root: $cz, key: $k, rmax: 20}, level: $lv, weight: {root: $w, key: $k}}}},
    holdout_file: $ho, holdout_mode: "mask", note: $note}' > "$OUT/source.json"
if [ "$DRY" != 1 ]; then
  kept=$(grep -o 'kept [0-9]* ([0-9.]* mm3)' "$OUT/logs/surflabels.log" | head -1)
  echo "$S pass $pass labels $kept weight $(grep -o 'weight sum kept [0-9.]*' "$OUT/logs/weight.log" | head -1) codes $(du -sh "$LB/codes.zarr" | cut -f1) weight $(du -sh "$OUT/weight.zarr" | cut -f1)" > "$OUT/summary.txt"
  [ "${KEEP:-0}" = 1 ] || { for d in "$LB/phase.zarr" "$GP/phase.zarr"; do case "$d" in "$OUT"/*/phase.zarr) rm -rf -- "$d";; esac; done; }
  log "done: $(cat "$OUT/summary.txt")"
fi

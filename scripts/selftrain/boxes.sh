#!/usr/bin/env bash
# boxes.sh: the round's box list per scan, A/boxes/<SCROLL>.json ([[z,y,x,nz,ny,nx], ...] in training-level voxels): NB (scans.tsv) random
# non-overlapping 1024^3 boxes with object fraction >= MINF (0.6) of the scanprep mask, origins on the 512 label-shard grid (--align 512),
# none within MARGIN_MM (0.5) of a bench holdout box of the same CT (HO, matched by the CT's basename; l0 voxels scaled to the level).
# Existing lists are kept (delete one to redraw it). Env: A, U (a binary with dino-boxes --align), MINF, ONLY (scroll list), HO, MARGIN_MM
set -u
A=${A:-/vesuvius/ufsm/selftrain}; U=${U:-$A/src/build/ufsm}; D=$(dirname "$(readlink -f "$0")"); mkdir -p $A/boxes
HO=${HO:-/vesuvius/ufsm/wbench/holdouts.json}; [ -s $HO ] || { echo "boxes.sh: no holdouts $HO"; exit 1; }
awk -F'\t' '!/^#/ && $8 == "go"' $D/scans.tsv | while IFS=$'\t' read -r S SC LV UM VOL AX NB ST; do
  [ -n "${ONLY:-}" ] && ! [[ " $ONLY " == *" $S "* ]] && continue
  [ -s $A/boxes/$S.json ] && continue
  M=/vesuvius/ufsm/u0/prep/$S-$SC.mask.raw; [ -s $M ] || { echo "$S: no mask $M"; continue; }
  $U dino-boxes --mask $M --out-level $LV --block 1024 --top $((NB * 4)) --min-frac ${MINF:-0.6} --align 512 > $A/boxes/$S.all.json 2> $A/boxes/$S.log \
    || { echo "$S: dino-boxes failed: $(tail -1 $A/boxes/$S.log)"; continue; }
  # holdouts of this CT in level-LV voxels, grown by the margin (UM is the level-LV voxel size)
  mg=$(awk -v m=${MARGIN_MM:-0.5} -v u=$UM 'BEGIN {print int(m * 1000 / u + 0.999)}')
  jq --arg b "$(basename $VOL)" --argjson lv $LV --argjson mg $mg --argjson nb $NB '
    [input.boxes[] | select((.ct | split("/") | last) == $b) | .l0 | map(. / pow(2; $lv))] as $h
    | [ .[] | . as $x | select(all($h[]; . as $o | any(range(3); ($x[.] + $x[. + 3] + $mg <= $o[.]) or ($o[.] + $o[. + 3] + $mg <= $x[.])))) ] | .[:$nb]' \
    $A/boxes/$S.all.json $HO > $A/boxes/$S.json.tmp && mv $A/boxes/$S.json.tmp $A/boxes/$S.json \
    && echo "$S: $(jq length $A/boxes/$S.json) boxes of $(jq length $A/boxes/$S.all.json) drawn, $(jq --arg b "$(basename $VOL)" '[.boxes[] | select((.ct | split("/") | last) == $b)] | length' $HO) holdouts"
done

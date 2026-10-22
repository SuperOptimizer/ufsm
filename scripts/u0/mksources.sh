#!/usr/bin/env bash
# scripts/u0/mksources.sh LIST OUT.json [TEST.json]: the multi-scroll sources file of every prepped scan in LIST (jq, no Python).
# Per source: the local root, level L only (min_level = max_level = L: the ~9.6 um level), "ct_only": 1 (no targets), the scanprep
# mask (windows around valid window centres), "norm" = scanprep median / scale (used only by sources that keep it: the sampler's
# default is the window z-score when "norm" is absent), "iso" = the voxel Otsu cut (papyrus / void), weight = valid window centres /
# number of scans of that scroll (a scroll with 7 scans does not get 7x the draws). PHerc1667 (the held-out test scroll) goes only to
# TEST.json. The cell count is in 16^3 level-L cells (~154 um at 9.6 um).
set -u
LIST=${1:?inventory rows}; OUT=${2:?out.json}; TEST=${3:-${OUT%.json}-test.json}
A=${A:-/vesuvius/ufsm/u0}; OLD=${OLD:-/vesuvius/ufsm/volcomp}
rows=$(mktemp)
while IFS=$'\t' read -r scroll id volume um kev dist level lum shape lgb cgb shards status newgb pairs; do
  [ "$scroll" = scroll ] && continue
  tag=$scroll-$id; J=$A/prep/$tag.json
  [ -e $A/state/$tag.prep.done ] && [ -e $J ] || continue
  case "$status" in use-*) root=$OLD/$scroll; key=$volume;; *) root=$A/vol; key=$scroll/$volume;; esac
  jq -c --arg name "$tag" --arg scroll "$scroll" --arg root "$root" --arg key "$key" --argjson um "$um" --argjson lev "$level" --arg kev "$kev" \
    '{name: $name, scroll: $scroll, root: $root, ct: $key, um: $um, min_level: $lev, max_level: $lev, ct_only: 1, keV: ($kev | tonumber? // null),
      mask: {map: .mask, min: 2}, norm: {center: .stats.median, scale: .stats.scale}, iso: .stats.otsu, valid: .valid_cells}' $J >> $rows
done < "$LIST"
jq -s '(group_by(.scroll) | map({key: .[0].scroll, value: length}) | from_entries) as $n
       | map(. + {weight: ((.valid / $n[.scroll]) | floor)})' $rows > $rows.all
jq '{sources: map(select(.scroll != "PHerc1667") | del(.scroll, .valid))}' $rows.all > $OUT
jq '{sources: map(select(.scroll == "PHerc1667") | del(.scroll, .valid))}' $rows.all > $TEST
echo "$(jq '.sources | length' $OUT) training sources -> $OUT, $(jq '.sources | length' $TEST) test -> $TEST"
rm -f $rows $rows.all

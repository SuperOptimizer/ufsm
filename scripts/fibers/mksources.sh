#!/usr/bin/env bash
# scripts/fibers/mksources.sh [OUT.json]: the sources of the H / V fibre mask model: one source per scroll with finished fibre labels
# (scripts/fibers/mklabels.sh: $LAB/<scroll>/done), CT level 0 from scrolls.tsv, target "recto" = the H plane, "sheet" = the V plane
# (prob254: 254 the family's fibres, 0 the other family's and air, 255 unknown = no loss), weights by labelled bytes, wbench holdouts.
# jq only.
set -euo pipefail
O=${1:-/vesuvius/ufsm/fibers/sources-fibhv.json}; LAB=${LAB:-/vesuvius/ufsm/fibers/labels}; TSV=/vesuvius/ufsm/v2/scrolls.tsv; CTL=/vesuvius/ufsm/v2/ct
T=$(mktemp); echo '[]' > $T
for d in $LAB/*/; do S=$(basename $d); [ -e $d/done ] || continue
  row=$(awk -F'\t' -v s=$S '$1 == s' $TSV); [ -n "$row" ] || continue
  IFS=$'\t' read -r _ _ _ _ _ _ _ AX _ CTK UM LV _ <<< "$row"
  by=$(du -sb $d/H | cut -f1)
  jq --arg n "$S-fib" --arg ct "$CTK" --argjson um $UM --arg ax "$AX" --arg h "$d/H" --arg v "$d/V" --arg root "$CTL" --argjson by $by \
     '. + [{name: $n, scope: "local", root: $root, ct: $ct, um: $um, min_level: 0, max_level: 0, axis: $ax, iso: 64,
            holdout_file: "/vesuvius/ufsm/wbench/holdouts.json", bytes: $by,
            targets: {recto: {root: $h, group: ".", encoding: "prob254"}, sheet: {root: $v, group: ".", encoding: "prob254"}}}]' $T > $T.n && mv $T.n $T
done
jq -n --slurpfile c $T '($c[0] | map(.bytes) | add // 1) as $tot | {cache: "/vesuvius/ufsm/cache", sources: ($c[0] | map(.weight = (.bytes / $tot) | del(.bytes)))}' > $O
rm -f $T; jq -r '.sources[] | "\(.name)\t\(.weight)"' $O

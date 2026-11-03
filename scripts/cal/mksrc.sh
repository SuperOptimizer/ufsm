#!/usr/bin/env bash
# mksrc.sh [OUT] (CAL 11-01, POD-L): calibration sources = published-segment GT of P4 June + 0172 + 0814 + 0500P2, weights per SCROLL
# (balanced, not by labelled volume: no scroll's density bias dominates), CT streamed from the volcomp mirror into a local cache
# (prefetch first). 0139 / 1667 never appear (gate scrolls). P4 wbench holdouts (holdouts.json) + 0.5 mm margin dummies.
# Env: W_P4 W_172 W_814 W_500 (.35 .35 .25 .05).
set -euo pipefail; C=/root/cal; OUT=${1:-$C/sources-cal.json}; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
HO=$C/misc/holdouts.json; T=$(mktemp -d); trap 'rm -rf $T' EXIT
P4CT=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr; G=$C/gt
# P4 codes are on the 9.6 grid (key 9.6); the band needs a 4.8 shape reference (metadata only, as desk p4.sh)
if [ -d $G/PHercParis4-wind/sets/codesA/9.6 ] && [ ! -e $G/PHercParis4-wind/sets/codesA/4.8/zarr.json ]; then
  mkdir -p $G/PHercParis4-wind/sets/codesA/4.8
  cp $C/misc/p4codes48.json $G/PHercParis4-wind/sets/codesA/4.8/zarr.json
fi
src() {   # src NAME CTKEY UM LEVELSPEC CODES CKEY PHASE PKEY PLEVEL AXIS TRUST WEIGHT RAD SPAN
  jq -nc --arg n $1 --arg r $M --arg c $2 --argjson um $3 --argjson lv "$4" --arg cr $5 --arg ck $6 --arg pr $7 --arg pk $8 --argjson pl $9 \
    --arg ax ${10} --argjson tb ${11} --argjson w ${12} --argjson rad ${13} --argjson span ${14} \
    '{name: $n, root: $r, ct: $c, um: $um} + $lv + {targets: {band: {root: $cr, key: $ck, radius: $rad, span: $span, phase: {root: $pr, key: $pk, level: $pl}}},
      axis: $ax, trust_band: $tb, weight: $w}'; }
{
  [ -e $G/PHercParis4-wind/DONE ] && src PHercParis4-june $P4CT 2.4 '{"min_level":2,"max_level":2}' $G/PHercParis4-wind/sets/codesA 4.8 $G/PHercParis4-wind/sets/phaseA 9.6 2 $C/axis/paris4-axis-0.json 20 ${W_P4:-.35} 80 75
  : # (11-03: phase one level coarser than the training level; needs a native-res rebuild) [ -e $G/PHerc0172-wind/DONE ] && src PHerc0172-wind PHerc0172/volumes/20241024131839-7.910um-53keV-masked.zarr 7.91 '{"min_level":0,"max_level":0}' $G/PHerc0172-wind/codes 15.82 $G/PHerc0172-wind/phase 15.82 1 $C/axis/PHerc0172.json 6 ${W_172:-.35} 24 22
  : # (11-03: phase one level coarser than the training level; needs a native-res rebuild) src PHerc0814-wind PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr 9.362 '{"min_level":0,"max_level":0}' $G/PHerc0814-wind-20261019/codes 18.724 $G/PHerc0814-wind-20261019/phase 18.724 1 $C/axis/PHerc0814-20250804134230.json 5 ${W_814:-.25} 20 19
  : # (11-03: phase one level coarser than the training level; needs a native-res rebuild) src PHerc0500P2-wind PHerc0500P2/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr 9.362 '{"min_level":0,"max_level":0}' $G/PHerc0500P2-wind-20261019/codes 18.724 $G/PHerc0500P2-wind-20261019/phase 18.724 1 $C/axis/PHerc0500P2-20250820143440.json 5 ${W_500:-.05} 20 19
} > $T/all.jsonl
# wbench holdouts: margin dummies (weight 1e-6) for every box on a CT some source uses (the sampler holds them out of every source of that CT)
jq -c '.boxes[]' $HO | while read -r b; do c=$(jq -r '.ct | split("/") | last' <<< "$b")
  s=$(jq -c --arg c "$c" 'select(.ct | split("/") | last == $c)' $T/all.jsonl | head -1); [ -n "$s" ] || continue
  jq -nc --argjson b "$b" --argjson s "$s" '(500 / $s.um | ceil) as $m | $s + {name: ("hold-" + $b.name), weight: 0.000001,
    holdout: [$b.l0[0] - $m, $b.l0[1] - $m, $b.l0[2] - $m, $b.l0[3] + 2 * $m, $b.l0[4] + 2 * $m, $b.l0[5] + 2 * $m]}'; done > $T/hold.jsonl
cat $T/all.jsonl $T/hold.jsonl | jq -c --arg h $HO '. + {holdout_file: $h}' | jq -s --arg c $C/cache '{cache: $c, sources: .}' > $OUT
# labels at the training level (11-03 rule; train refuses anything coarser): every stored phase on the grid of the source's one training level
jq -e '[.sources[] | select(.targets.band.phase) | select(.min_level != .targets.band.phase.level or .max_level != .targets.band.phase.level) | .name] as $b
  | if ($b | length) > 0 then error("stored phase not at the training level: \($b | join(", "))") else true end' $OUT > /dev/null
jq -r '.sources[] | select(.weight > 0.001) | "\(.name) w \(.weight) um \(.um) phase \(.targets.band.phase.key)"' $OUT; echo "$(jq '.sources | length' $OUT) sources -> $OUT"

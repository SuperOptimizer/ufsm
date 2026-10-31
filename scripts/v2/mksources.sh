#!/usr/bin/env bash
# scripts/v2/mksources.sh OUT.json: the v2 universal-model sources (winding + ink), modelled on scripts/selftrain/mksources.sh.
#  1. surface sources, one per train scroll of V2/scrolls.tsv with V2LABEL labels (V2/labels/<S>/phase.zarr/<KEY>: pair-rule band phase on
#     the scan's training grid, ignored voxels already 255): phase-only band target (radius 20, span 19, stored phase at the scan's
#     training level: 0 for the 8.64 / 9.36 um scans, 2 for the 2.4 um scans) + targets.ink = the pseudo ink (V2/ink/pseudo/<S>.zarr,
#     prob254, ift3-24000 in the recto band; "weight" PINKW 0.5 = its loss weight relative to human ink) when it exists
#     + "focus" (phase-packing --map of the labels, w FOCUSW 2: packed chunks drawn more often). Weights by labelled mm3
#     (phase-packing's labelled voxels with neighbours x um^3), sharing SURF (0.60) of the total.
#  2. Paris 4 GT (PK8: june / july phaseA/B + its PH1-3 / HOLD holdout dummies, CT root -> V2/ct): GTSHARE (0.25) of the total, so the
#     published GT outweighs any single surface source (high weight).
#  3. human ink sources (scripts/v2/inkhuman.tsv = ift3's tight human ink-splat sets, V2/ink/human/tight; 0139 / 1667 left out):
#     targets.ink prob254, no phase target, sharing INKSHARE (0.15) with ift3's relative weights. CT local (V2/ct) when fetched, else the mirror.
#  4. holdouts: "holdout_file" HO (wbench holdouts.json) on every source + weight-1e-6 copies holding each box grown by MARGIN_MM (0.5)
#     (the sampler holds a source's boxes out of every source of the same CT). Whole-scroll holdouts (1667, 0139, 1299, 0846A) never
#     appear: scrolls.tsv marks them, inkhuman.tsv leaves them out.
# The per-source table goes to stderr. Env: V2 (/vesuvius/ufsm/v2), U (V2/src/build/ufsm), HO, PK8, SURF / GTSHARE / INKSHARE, PINKW,
# FOCUSW, MARGIN_MM, MINMM3 (1: surface scrolls with fewer labelled mm3 are left out), TH (16).
set -euo pipefail
OUT=${1:?usage: mksources.sh OUT.json}; H=$(cd "$(dirname "$0")" && pwd)
V2=${V2:-/vesuvius/ufsm/v2}; U=${U:-$V2/src/build/ufsm}; TH=${TH:-16}
HO=${HO:-/vesuvius/ufsm/wbench/holdouts.json}; PK8=${PK8:-/vesuvius/ufsm/paris4/wnew/sources-pk8.json}
MIRROR=${MIRROR:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
SURF=${SURF:-0.60}; GTS=${GTSHARE:-0.25}; INKS=${INKSHARE:-0.15}; PINKW=${PINKW:-0.5}; FOCUSW=${FOCUSW:-2}
T=$(mktemp -d); trap 'rm -rf $T' EXIT; mkdir -p $V2/focus
# 1. surface sources: labelled volume + focus map per scroll (cached in V2/focus/<S>.{raw,raw.json,txt})
tail -n +2 $V2/scrolls.tsv | while IFS=$'\t' read -r S SPLIT NW WR WC SJ SL AX CR K UM LV FL PK _; do
  [ "$SPLIT" = train ] || continue
  L=$V2/labels/$S/phase.zarr; [ -d $L ] || { echo "mksources: $S: no labels ($L), left out" >&2; continue; }
  key=$(ls $L | grep -v '\.json$' | head -1)
  F=$V2/focus/$S
  if [ ! -s $F.txt ] || [ $L/$key/zarr.json -nt $F.txt ]; then
    read -r fz fy fx < <(jq -r '.shape | map(tostring) | join(" ")' $L/$key/zarr.json)
    $U phase-packing $L $key --um $key --box 0,0,0,$fz,$fy,$fx --map $F.raw --threads $TH > $F.txt 2>&1 || { echo "mksources: $S: phase-packing failed ($F.txt)" >&2; continue; }
  fi
  nk=$(sed -nE 's/.*: ([0-9]+) labelled voxels with neighbours.*/\1/p' $F.txt | head -1)
  mm3=$(awk -v n=${nk:-0} -v u=$key 'BEGIN { printf "%.2f", n * u * u * u * 1e-9 }')
  [ "$CR" != - ] || { echo "mksources: $S: CT not fetched yet, left out" >&2; continue; }
  ink=-; [ -e $V2/ink/pseudo/$S.done ] && ink=$V2/ink/pseudo/$S.zarr
  echo "$S $CR $K $UM $LV $AX $key $mm3 $ink" >> $T/surf.txt
done
[ -s $T/surf.txt ] || { echo "mksources: no labelled train scroll yet" >&2; exit 1; }
SV=$(awk -v m=${MINMM3:-1} '$8 >= m { s += $8 } END { print s + 0 }' $T/surf.txt)
printf "%-14s %10s %8s %s\n" scroll mm3 weight pseudo-ink >&2
while read -r S CR K UM LV AX key mm3 ink; do
  awk -v m=$mm3 -v t=${MINMM3:-1} 'BEGIN { exit !(m >= t) }' || { printf "%-14s %10s (left out: < MINMM3)\n" $S $mm3 >&2; continue; }
  w=$(awk -v m=$mm3 -v v=$SV -v s=$SURF 'BEGIN { printf "%.6f", s * m / v }')
  printf "%-14s %10s %8s %s\n" $S $mm3 $w $ink >&2
  jq -nc --arg n "$S-v2surf" --arg r $CR --arg c $K --argjson um $UM --argjson lv $LV --arg ax $AX --argjson w $w --arg pr $V2/labels/$S/phase.zarr \
     --arg pk $key --arg ink $ink --argjson iw $PINKW --arg fm $V2/focus/$S.raw --argjson fw $FOCUSW \
    '{name: $n, root: $r, ct: $c, um: $um, weight: $w} + (if $ax != "-" then {axis: $ax} else {} end)
     + (if $lv == 2 then {min_level: 2, max_level: 2} else {min_level: 0, max_level: 0} end)
     + {targets: ({band: {radius: 20, span: 19, phase: {root: $pr, key: $pk, level: $lv}}}
                  + (if $ink != "-" then {ink: {root: $ink, group: ".", encoding: "prob254", min_level: $lv, weight: $iw}} else {} end)),
        focus: {map: $fm, w: $fw}}' >> $T/surf.jsonl
done < $T/surf.txt
# 2. Paris 4 GT (relative weights kept, scaled to GTSHARE; CT from the v2 tree)
P4K=$(jq -r '.sources[0].ct' $PK8)
[ -e $V2/ct/$P4K ] || { echo "mksources: $V2/ct/$P4K missing" >&2; exit 1; }
GW=$(jq '[.sources[].weight] | add' $PK8)
jq -c --arg r $V2/ct --argjson s $(awk -v g=$GTS -v w=$GW 'BEGIN { print g / w }') '.sources[] | .root = $r | .weight = (.weight * $s)' $PK8 > $T/gt.jsonl
# 3. human ink (ift3's relative weights, scaled to INKSHARE)
IW=$(awk -F'\t' '!/^#/ { s += $6 } END { print s }' $H/inkhuman.tsv)
awk -F'\t' '!/^#/' $H/inkhuman.tsv | while IFS=$'\t' read -r n S vol um lv w lab hold ax; do
  r=$V2/ct; k=$S/$vol; [ -e $r/$k/$lv/zarr.json ] || { r=$MIRROR; k=$S/volumes/$vol; }
  jq -nc --arg n "$n" --arg r $r --arg c $k --argjson um $um --argjson lv $lv --argjson w $(awk -v w=$w -v i=$IW -v s=$INKS 'BEGIN { printf "%.6f", s * w / i }') \
     --arg lab $V2/ink/human/tight/$lab --arg hold "$hold" --arg ax "$ax" \
    '{name: $n, root: $r, ct: $c, um: $um, weight: $w, min_level: $lv, max_level: $lv} + (if $ax != "-" then {axis: $ax} else {} end)
     + (if $hold != "-" then {holdout: ($hold | split(",") | map(tonumber))} else {} end)
     + {targets: {ink: {root: $lab, group: ".", encoding: "prob254", min_level: $lv}}}' >> $T/ink.jsonl
done
# 4. bench holdouts: holdout_file everywhere + margin dummies per box of a CT some source uses
mg=${MARGIN_MM:-0.5}
cat $T/gt.jsonl $T/surf.jsonl $T/ink.jsonl > $T/all.jsonl
jq -c '.boxes[]' $HO | while read -r b; do
  c=$(jq -r '.ct | split("/") | last' <<< "$b")
  src=$(jq -c --arg c "$c" 'select(.ct | split("/") | last == $c) | select(.name | test("hold") | not)' $T/all.jsonl | head -1); [ -n "$src" ] || continue
  jq -nc --argjson b "$b" --argjson s "$src" --argjson mg $mg '($mg * 1000 / $s.um | ceil) as $m |
    $s + {name: ("hold-" + $b.name), weight: 0.000001,
          holdout: [$b.l0[0] - $m, $b.l0[1] - $m, $b.l0[2] - $m, $b.l0[3] + 2 * $m, $b.l0[4] + 2 * $m, $b.l0[5] + 2 * $m]}' >> $T/hold.jsonl
done
cat $T/all.jsonl $( [ -s $T/hold.jsonl ] && echo $T/hold.jsonl ) | jq -c --arg h $HO '. + {holdout_file: $h}' |
  jq -s --arg c $V2/cache '{cache: $c, sources: .}' > $OUT
echo "mksources: $OUT: $(jq '.sources | length' $OUT) sources: $(wc -l < $T/surf.jsonl) surface ($SURF, $SV mm3), Paris 4 GT ($GTS), $(wc -l < $T/ink.jsonl) human ink ($INKS), $( [ -s $T/hold.jsonl ] && wc -l < $T/hold.jsonl || echo 0) holdout dummies" >&2

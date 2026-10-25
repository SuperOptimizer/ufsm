#!/usr/bin/env bash
# mksources.sh ARM OUT.json: the student's sources = GT (Paris 4 June / July from sources-pk8 with its PH1-3 / HOLD holdouts and packing
# focus; PHerc0139 published-wrap GT, level-1 stored phase) + one phase-only pseudo-label source per scan of round R (A/r$R/labels/ARM/<S>,
# fit --winding-mod14 3 codes on the scan's training level). Pseudo sources share PSHARE (0.35) of the total weight, split by labelled
# volume (mm^3 = gated voxels from A/r$R/stats/*.ARM.txt); GT keeps its relative weights. Bench holdouts (HO, all scrolls
# boxes): "holdout_file" on every source + weight-1e-6 copies holding each box grown by MARGIN_MM 0.5 (the sampler holds a source's box out of every source of
# that CT). PHerc0814 GT is not used (fit-oracle ceiling 0.2-0.3). A per-scan table goes to stderr.
# Env: A, R (0), PSHARE, HO, MARGIN_MM, PK8 (the Paris 4 sources), MINMM3 (0.5: scans with less pseudo volume are left out), EXCLUDE ("PHerc0814": scans without pseudo labels)
set -euo pipefail
ARM=$1 OUT=$2; A=${A:-/vesuvius/ufsm/selftrain}; R=${R:-0}; RD=$A/r$R; D=$(dirname "$(readlink -f "$0")")
HO=${HO:-/vesuvius/ufsm/wbench/holdouts.json}; PK8=${PK8:-/vesuvius/ufsm/paris4/wnew/sources-pk8.json}; CT=/vesuvius/ufsm/u0/vol
GT=/vesuvius/ufsm/gt; T=$(mktemp -d); trap 'rm -rf $T' EXIT
# GT: Paris 4 (all pk8 entries, holdout dummies included) + 0139
jq -c '.sources[]' $PK8 > $T/gt.jsonl
jq -nc --arg r $CT --arg g $GT/PHerc0139-wind-20261016 '{name: "PHerc0139-wind", root: $r, ct: "PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr",
  um: 9.362, max_level: 0, axis: "/vesuvius/ufsm/gt/axis/PHerc0139-umb-scan.json", trust_band: 5, weight: 0.0958,
  targets: {band: {root: ($g + "/codes"), key: "18.72", radius: 20, span: 19, phase: {root: ($g + "/phase"), key: "18.72", level: 1}}}}' >> $T/gt.jsonl
GW=$(jq -s 'map(.weight) | add' $T/gt.jsonl)
# pseudo: labelled volume per scan
awk -F'\t' '!/^#/ && $8 == "go"' $D/scans.tsv | while IFS=$'\t' read -r S SC LV UM VOL AX NB ST; do
  case " ${EXCLUDE-PHerc0814} " in *" $S "*) continue;; esac   # (PHerc0814: gated label precision 0.57 vs its GT)
  [ -d $RD/labels/$ARM/$S ] || continue
  n=$(cat $RD/stats/$S-b*.$ARM.txt 2>/dev/null | sed -nE 's/.*winding-gate [^:]*: ([0-9]+) of ([0-9]+) voxels.*/\1 \2/p' | awk '{a += $1; b += $2} END {print a + 0, b + 0}')
  read -r nl nt <<< "$n"
  mm3=$(awk -v n=$nl -v u=$UM -v l=$LV 'BEGIN {u *= (l == 2 ? 4 : 1); printf "%.2f", n * u * u * u * 1e-9}')
  key=$(ls $RD/labels/$ARM/$S | grep -v zarr.json | head -1)
  echo "$S $SC $LV $UM $VOL $AX $mm3 $nl $nt $key" >> $T/ps.txt
done
[ -s $T/ps.txt ] || { echo "mksources: no pseudo labels for arm $ARM" >&2; exit 1; }
PV=$(awk -v m=${MINMM3:-0.5} '$7 >= m {s += $7} END {print s + 0}' $T/ps.txt)
PS=${PSHARE:-0.35}; PW=$(awk -v g=$GW -v p=$PS 'BEGIN {print g * p / (1 - p)}')
printf "%-14s %8s %12s %7s %s\n" scan mm3 voxels share weight >&2
while read -r S SC LV UM VOL AX mm3 nl nt key; do
  awk -v m=$mm3 -v t=${MINMM3:-0.5} 'BEGIN { exit !(m >= t) }' || { printf "%-14s %8s %12s %6.1f%% (left out)\n" $S $mm3 $nl $(awk -v a=$nl -v b=$nt 'BEGIN {print b ? 100 * a / b : 0}') >&2; continue; }
  w=$(awk -v m=$mm3 -v v=$PV -v p=$PW 'BEGIN {printf "%.6f", p * m / v}')
  printf "%-14s %8s %12s %6.1f%% %s\n" $S $mm3 $nl $(awk -v a=$nl -v b=$nt 'BEGIN {print b ? 100 * a / b : 0}') $w >&2
  jq -nc --arg n "$S-pseudo-r$R-$ARM" --arg r $CT --arg c $VOL --argjson um $UM --argjson lv $LV --arg ax $AX --argjson w $w --arg pr $RD/labels/$ARM/$S --arg pk $key \
    '{name: $n, root: $r, ct: $c, um: $um, axis: $ax, weight: $w} + (if $lv == 2 then {min_level: 2} else {max_level: 0} end)
     + {targets: {band: {radius: 20, span: 19, phase: {root: $pr, key: $pk, level: $lv}}}}' >> $T/ps.jsonl
done < $T/ps.txt
# bench holdouts (HO, every scroll incl. Paris 4's July-only slabs): every source carries "holdout_file" (the sampler holds out the
# boxes of its own CT); plus, per box of a CT some source uses, a weight-1e-6 copy of that source whose holdout is the box grown by
# MARGIN_MM (the sampler holds a source's boxes out of every source of the same CT)
mg=${MARGIN_MM:-0.5}
cat $T/gt.jsonl $T/ps.jsonl > $T/all.jsonl
jq -c '.boxes[]' $HO | while read -r b; do
  c=$(jq -r '.ct | split("/") | last' <<< "$b")
  src=$(jq -c --arg c "$c" 'select(.ct | split("/") | last == $c)' $T/all.jsonl | head -1); [ -n "$src" ] || continue
  jq -nc --argjson b "$b" --argjson s "$src" --argjson mg $mg '($mg * 1000 / $s.um | ceil) as $m |
    $s + {name: ("hold-" + $b.name), weight: 0.000001,
          holdout: [$b.l0[0] - $m, $b.l0[1] - $m, $b.l0[2] - $m, $b.l0[3] + 2 * $m, $b.l0[4] + 2 * $m, $b.l0[5] + 2 * $m]}' >> $T/hold.jsonl
done
cat $T/all.jsonl $( [ -s $T/hold.jsonl ] && echo $T/hold.jsonl ) | jq -c --arg h $HO '. + {holdout_file: $h}' | jq -s '{cache: "/vesuvius/ufsm/cache", sources: .}' > $OUT
echo "mksources: $OUT: $(jq '.sources | length' $OUT) sources, GT weight $GW, pseudo weight $PW (share $PS) over $PV mm3" >&2

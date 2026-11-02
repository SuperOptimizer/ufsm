#!/usr/bin/env bash
# scripts/rv/mksources.sh [OUT.json]: sources of the recto / verso side model (train --task unwrap2 --rv 1, scripts/rv/train.sh).
# Every source is phase-only (targets.band.phase: the pair-rule winding phase; the sampler's papyrus plane = CT >= "iso" gives
# recto side = papyrus with phase < 1/2 turn, verso the outer half). Training grid ~8-9.6 um (each scan's 8.64 / 9.36 um level 0,
# or level 2 of the 2.4 um scans).
#  1. published GT (GTSHARE of the weight, 0.6): Paris 4 June / July phaseA / phaseB (gt/paris4-wind-20261019, 9.6 = level 2),
#     PHerc0139 (gt/PHerc0139-wind-20261016/phase 18.72 = level 1 of the 9.362 um scan), and whatever GTX lists
#     (lines "name ct_root ct_key um level axis phase_root phase_key phase_level weight", e.g. 0172 / 0500P2 once checked).
#     NEVER 0814 (skips sheets) or 1667 (the eval scroll).
#  2. our surfaces (SURFSHARE, 0.4): V2/labels/<S>/phase.zarr of scripts/v2/surflabels.sh (gated), one source per scroll in LABS
#     (default: every V2/labels/<S>/done of a train scroll), weighted by labelled mm3 (LABW env "S=w ..." multiplies).
#  3. holdouts: every wbench holdouts.json box grown by MARGIN_MM (0.5) as a weight-1e-6 dummy of its CT (the sampler holds a
#     source's box out of every source of the same CT), plus "holdout_file" on every source.
# Env: V2 (/vesuvius/ufsm/v2), ISO (64: papyrus cutoff), HO, MARGIN_MM, GTSHARE, SURFSHARE, GTX, LABS, CTR (CT root /vesuvius/ufsm/v2/ct).
set -euo pipefail
OUT=${1:-/vesuvius/ufsm/v2/rv/sources-rv.json}
V2=${V2:-/vesuvius/ufsm/v2}; ISO=${ISO:-64}; HO=${HO:-/vesuvius/ufsm/wbench/holdouts.json}; MM=${MARGIN_MM:-0.5}
GTS=${GTSHARE:-0.6}; SFS=${SURFSHARE:-0.4}; CTR=${CTR:-/vesuvius/ufsm/v2/ct}; GT=/vesuvius/ufsm/gt
T=$(mktemp -d); trap 'rm -rf $T' EXIT
# name ct_root ct_key um level axis phase_root phase_key phase_level weight
cat > $T/gt.txt <<G
PHercParis4-june $CTR PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr 2.4 2 /vesuvius/ufsm/teacher/axis-0.json $GT/paris4-wind-20261019/phaseA 9.6 2 1
PHercParis4-july $CTR PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr 2.4 2 /vesuvius/ufsm/teacher/axis-0.json $GT/paris4-wind-20261019/phaseB 9.6 2 0.559
PHerc0139-gt $CTR PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr 9.362 0 /mnt/scratch/pods/PHerc0139/unroll/axis.json $GT/PHerc0139-wind-20261016/phase 18.72 1 0.6
G
[ -n "${GTX:-}" ] && grep -v '^#' $GTX >> $T/gt.txt
GW=$(awk '{s += $10} END {print s}' $T/gt.txt)
src() {   # name root key um level axis proot pkey plevel weight scope
  jq -nc --arg n $1 --arg r $2 --arg c $3 --argjson um $4 --argjson lv $5 --arg ax $6 --arg pr $7 --arg pk $8 --argjson pl $9 --argjson w ${10} \
     --argjson iso $ISO --arg ho $HO --arg sc ${11} \
    '{name: $n, scope: $sc, root: $r, ct: $c, um: $um, min_level: $lv, max_level: $lv, axis: $ax, iso: $iso, weight: $w, holdout_file: $ho,
      targets: {band: {radius: 20, span: 19, phase: {root: $pr, key: $pk, level: $pl}}}}'
}
while read -r n r c um lv ax pr pk pl w; do
  src $n $r $c $um $lv $ax $pr $pk $pl $(awk -v w=$w -v g=$GW -v s=$GTS 'BEGIN {printf "%.6f", s * w / g}') global >> $T/all.jsonl
done < $T/gt.txt
# 2. our surfaces
: > $T/surf.txt
for S in ${LABS:-$(ls $V2/labels 2>/dev/null)}; do
  L=$V2/labels/$S; [ -e $L/done ] && [ -d $L/phase.zarr ] || continue
  case $S in PHerc1667|PHerc0814|PHerc0139|PHerc1299|PHerc0846A) echo "mksources: $S is a holdout / invalid: left out" >&2; continue;; esac
  row=$(awk -F'\t' -v s=$S '$1 == s' $V2/scrolls.tsv); [ -n "$row" ] || continue
  IFS=$'\t' read -r _ split _ _ _ _ _ ax ctr ctk ctum ctlev _ <<< "$row"; [ "$split" = train ] || continue
  key=$(ls $L/phase.zarr | grep -v '\.json$' | head -1)
  mm3=$(sed -nE 's/.*kept [0-9]+ \(([0-9.]+) mm3\).*/\1/p' $L/surflabels.log | tail -1)
  lw=$(echo " ${LABW:-} " | sed -nE "s/.* $S=([0-9.]+) .*/\1/p"); lw=${lw:-1}
  echo "$S $CTR $ctk $ctum $ctlev $ax $L/phase.zarr $key $ctlev $(awk -v m=${mm3:-0} -v w=$lw 'BEGIN {print m * w}')" >> $T/surf.txt
done
SV=$(awk '{s += $10} END {print s + 0}' $T/surf.txt)
while read -r n r c um lv ax pr pk pl w; do
  src $n-surf $r $c $um $lv $ax $pr $pk $pl $(awk -v w=$w -v g=$SV -v s=$SFS 'BEGIN {printf "%.6f", g > 0 ? s * w / g : 0}') local >> $T/all.jsonl
done < $T/surf.txt
# 3. holdout dummies: each box grown by MM mm, on the source of its CT (any source with that CT basename)
jq -c '.boxes[]' $HO | while read -r b; do
  ct=$(jq -r .ct <<< "$b"); m=$(jq -c --arg c "$(basename $ct)" 'select((.ct | split("/") | last) == $c)' $T/all.jsonl | head -1)
  [ -n "$m" ] || continue
  um0=$(jq -r .um <<< "$m"); g=$(awk -v mm=$MM -v u=$um0 'BEGIN {printf "%d", mm * 1000 / u}')
  jq -c --argjson b "$b" --argjson g $g '.name = (.name + "-hold-" + $b.name) | .weight = 0.000001
    | .holdout = [($b.l0[0] - $g), ($b.l0[1] - $g), ($b.l0[2] - $g), ($b.l0[3] + 2 * $g), ($b.l0[4] + 2 * $g), ($b.l0[5] + 2 * $g)]' <<< "$m" >> $T/hold.jsonl
done
cat $T/all.jsonl $T/hold.jsonl 2>/dev/null | jq -s '{cache: "/vesuvius/ufsm/cache", sources: .}' > $OUT
jq -r '.sources[] | "\(.name)\t\(.weight)\t\(.scope)\t\(.holdout // "")"' $OUT >&2

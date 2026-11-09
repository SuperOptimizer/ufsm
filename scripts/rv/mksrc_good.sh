#!/usr/bin/env bash
# scripts/rv/mksrc_good.sh BASE.json OUT.json [WC]: BASE (the GT rv sources, e.g. sources-rv-gate.json) + one source per finished
# scripts/rv/good.sh scroll ($GOOD/S/done): CT from scrolls.tsv (local /vesuvius/ufsm/v2/ct), phase $GOOD/S/phase.zarr (gated cells
# already unknown, no separate ignore), scope "local" (our turn counts: no long-range winding mod 4), iso 64, holdouts.json.
# Weights: BASE weights x (1 - WC); the corpus shares WC (default 0.4) in proportion to each scroll's gated-phase bytes (~ labelled volume).
# jq only (no Python).
set -euo pipefail
B=$1; O=$2; WC=${3:-0.4}; GOOD=${GOOD:-/mnt/scratch/rv/good}; TSV=/vesuvius/ufsm/v2/scrolls.tsv; CTL=/vesuvius/ufsm/v2/ct
HO=/vesuvius/ufsm/wbench/holdouts.json
T=$(mktemp); echo '[]' > $T
for d in $GOOD/*/; do S=$(basename $d); [ -e $d/done ] || continue
  row=$(awk -F'\t' -v s=$S '$1 == s' $TSV); [ -n "$row" ] || continue
  IFS=$'\t' read -r _ _ _ _ _ _ _ AX _ CTK UM LV _ <<< "$row"
  key=$(ls $d/phase.zarr | grep -v zarr.json | head -1); by=$(du -sb $d/phase.zarr | cut -f1)
  [ -e $CTL/$CTK/$LV/zarr.json ] || { echo "mksrc_good: $S no local CT $CTL/$CTK/$LV, skipped" >&2; continue; }
  jq --arg n "$S-good" --arg ct "$CTK" --argjson um $UM --argjson lv $LV --arg ax "$AX" --arg ph "$d/phase.zarr" --arg k "$key" \
     --arg ho "$HO" --arg root "$CTL" --argjson by $by \
     '. + [{name: $n, scope: "local", root: $root, ct: $ct, um: $um, min_level: $lv, max_level: $lv, axis: $ax, iso: 64, holdout_file: $ho,
            bytes: $by, targets: {band: {radius: 20, span: 19, phase: {root: $ph, key: $k, level: $lv}}}}]' $T > $T.n && mv $T.n $T
done
jq --argjson wc $WC --slurpfile c $T '
  ($c[0] | map(.bytes) | add // 0) as $tot
  | .sources |= (map(.weight *= (1 - $wc)) + ($c[0] | map(.weight = ($wc * .bytes / $tot) | del(.bytes))))' $B > $O
rm -f $T
jq -r '.sources[] | select(.weight > 0.001) | "\(.name)\t\(.weight)"' $O

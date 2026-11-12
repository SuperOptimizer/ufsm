#!/usr/bin/env bash
# scripts/latdino/ssl-sources.sh IN.json OUT.json [SHARE]: IN's sources plus one unlabelled ("ct_only": 1) source per CT volume in
# $CTROOT (default /vesuvius/ufsm/v2/ct) for train --ssl (DESIGN 2026-11-12 "Self-supervision for the latent net").
# - EXCL (default PHerc0175A): scrolls left out entirely (the strict unseen control).
# - Every box of $HOLD (default /vesuvius/ufsm/inkgt/holdouts-ink.json: the wbench boxes + the ink eval crops) on that CT rejects
#   training windows ("holdout_reject": all names): an unlabelled window has no labels to mask, so held-out CT is never drawn.
# - Levels as the labelled sources: scans finer than 3 um at levels 2..3 (9.6 / 19.2 um), the 7.9-9.4 um scans at 0..1.
# - The axis only where a labelled IN source on the same CT names one (its frame is known); otherwise none (the radial channels are
#   then constant, as for the u2ink sources without one).
# - SHARE is only echoed: pass it as train --ssl-share (the trainer rescales the ct_only weights; each gets weight 1 here).
set -euo pipefail
IN=${1:?IN.json}; OUT=${2:?OUT.json}; SHARE=${3:-0.25}
CTROOT=${CTROOT:-/vesuvius/ufsm/v2/ct}; HOLD=${HOLD:-/vesuvius/ufsm/inkgt/holdouts-ink.json}; EXCL=${EXCL:-PHerc0175A}
tmp=$(mktemp); trap 'rm -f $tmp' EXIT
n=0
for vol in "$CTROOT"/PH*/*.zarr; do
  scroll=$(basename "$(dirname "$vol")"); key=$scroll/$(basename "$vol")
  case " $EXCL " in *" $scroll "*) continue;; esac
  um=$(basename "$vol" | sed -n 's/^[0-9]*-\([0-9.]*\)um-.*/\1/p'); [ -n "$um" ] || { echo "skip $key: no voxel size in the name" >&2; continue; }
  if awk -v u="$um" 'BEGIN{exit !(u < 3)}'; then lo=2; hi=3; else lo=0; hi=1; fi
  base=$(basename "$vol")
  rej=$(jq -c --arg b "$base" '[.boxes[] | select((.ct | split("/") | last) == $b) | .name]' "$HOLD")
  ax=$(jq -r --arg k "$key" '[.sources[] | select(.ct == $k and (.ct_only // 0) == 0 and .axis != null) | .axis][0] // ""' "$IN")
  jq -n --arg name "$scroll-$um-u" --arg root "$CTROOT" --arg ct "$key" --argjson um "$um" --argjson lo $lo --argjson hi $hi \
        --arg hold "$HOLD" --argjson rej "$rej" --arg ax "$ax" \
        '{name: $name, scope: "global", root: $root, ct: $ct, um: $um, iso: 64, min_level: $lo, max_level: $hi, weight: 1, ct_only: 1,
          holdout_file: $hold, holdout_reject: $rej} + (if $ax != "" then {axis: $ax} else {} end)' >> $tmp
  n=$((n + 1))
done
jq --slurpfile u $tmp '.sources += $u | .note = ((.note // "") + " + ssl-sources.sh: \($u | length) ct_only volumes (excluded '"$EXCL"')")' "$IN" > "$OUT"
echo "$OUT: $(jq '.sources | length' "$OUT") sources, $n unlabelled; train with --ssl 16 --ssl-share $SHARE" >&2

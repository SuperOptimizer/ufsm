#!/usr/bin/env bash
# scripts/pseudo/mksources.sh BASE.json OUT.json W DIR...: BASE's sources plus one pseudo source per scripts/pseudo/build.sh output DIR
# (its source.json; DIRs whose summary says REFUSED, or without weight.zarr/done, are skipped and named). The pseudo sources share W
# (e.g. 0.15) of the total weight in proportion to each one's gated label volume (surf-labels' kept voxels x phase-coh's kept weight
# share); BASE's weights are scaled by (1 - W) so the total stays. jq only.
set -euo pipefail
B=${1:?BASE.json}; O=${2:?OUT.json}; W=${3:?W}; shift 3
T=$(mktemp); trap 'rm -f "$T"' EXIT; echo '[]' > "$T"
for d in "$@"; do
  if grep -q REFUSED "$d/summary.txt" 2>/dev/null || [ ! -e "$d/weight.zarr/done" ] || [ ! -s "$d/source.json" ]; then echo "mksources: $d skipped ($(cat "$d/summary.txt" 2>/dev/null || echo 'not built'))" >&2; continue; fi
  kv=$(grep -o 'kept [0-9]* (' "$d/logs/surflabels.log" | head -1 | tr -dc 0-9)
  ws=$(grep -o 'weight sum kept [0-9.]*' "$d/logs/weight.log" | head -1 | awk '{print $4}'); ws=${ws:-1}
  vol=$(awk -v k="${kv:-0}" -v w="$ws" 'BEGIN {printf "%.0f", k * w}')
  [ "$vol" -gt 0 ] || { echo "mksources: $d has no gated labels, skipped" >&2; continue; }
  jq --slurpfile s "$d/source.json" --argjson v "$vol" '. + [$s[0] + {vol: $v}]' "$T" > "$T.n" && mv "$T.n" "$T"
done
jq --argjson w "$W" --slurpfile c "$T" '
  ($c[0] | map(.vol) | add // 0) as $tot
  | if $tot == 0 then error("no pseudo source with labels") else . end
  | .sources |= (map(.weight *= (1 - $w)) + ($c[0] | map(.weight = ($w * .vol / $tot) | del(.vol))))
  | .note = ((.note // "") + " + scripts/pseudo/mksources.sh: \($c[0] | length) pseudo sources at total weight \($w)")' "$B" > "$O"
jq -r '.sources[] | select(.name | endswith("-pseudo")) | "\(.name)\t\(.weight)"' "$O"

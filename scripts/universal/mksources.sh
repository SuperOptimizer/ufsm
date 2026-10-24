#!/usr/bin/env bash
# scripts/universal/mksources.sh OUT.json: the universal run's sources (jq only): the labelled sources (labelled-desk.json: Paris 4 L2, 0139,
# 0814 -- winding codes + human ink-splat labels, holdout boxes) + every U0 scan prepped so far (scripts/u0/mksources.sh: ct_only, scanprep
# masks, the ~9.6 um level; PHerc1667, the test scroll, never). The U0 entries of the labelled scans (same frames) get the labelled
# holdouts (Paris 4 78 keV: its eval holdout too); PHerc1447's community-surface eval box is held out too.
set -eu
OUT=${1:?out.json}; D=$(cd "$(dirname "$0")" && pwd); A=${A:-/vesuvius/ufsm/u0}
T=$(mktemp -d)
bash $A/scripts/mksources.sh $A/inventory.tsv $T/u0.json $T/u0-test.json > $T/log 2>&1 || { cat $T/log; exit 1; }
jq '.sources |= map(
      if (.ct | test("PHerc0139/20250728140407")) then . + {holdout: [[3472, 3472, 1424, 736, 736, 736]]}
      elif (.ct | test("PHerc0814/20250804134230")) then . + {holdout: [[2448, 2448, 2960, 736, 736, 736]]}
      elif (.ct | test("PHerc1447/20250521151220")) then . + {holdout: [[12280, 2820, 4890, 480, 480, 480]]}
      elif (.ct | test("20260411134726-2.400um")) then . + {holdout: [[12032, 3968, 3456, 1280, 896, 1152]]}
      else . end)' $T/u0.json > $T/u0h.json
# only scans whose local copy (ROOT/CT/LEVEL/zarr.json) exists: inventory rows U0 skipped (duplicates, 45 um, DLS) are never copied
jq -c '.sources[]' $T/u0h.json | while IFS= read -r e; do
  f=$(jq -r '"\(.root)/\(.ct)/\(.min_level)/zarr.json"' <<< "$e")
  if [ -e "$f" ]; then echo "$e"; else echo "mksources: skipped (no local copy): $f" >&2; fi
done | jq -s '{sources: .}' > $T/u0ok.json
# XSCAN=1: + the Paris 4 cross-scan views (labelled-xscan.json: 137 keV L2 and 1.13 um L3, labels warped from 78 keV by ufsm reg warp over
# the certified probe cells, the eval holdout's image excluded from their boxes)
X=$D/labelled-desk.json; [ "${XSCAN:-0}" = 1 ] && { jq -s '{sources: (.[0].sources + .[1].sources)}' $D/labelled-desk.json $D/labelled-xscan.json > $T/lab.json; X=$T/lab.json; }
# PSEUDO=DIR: + the wind96 pseudo-label boxes (pseudo-sources.sh: same-scan U0 frames, holdouts respected), the pseudo group's total weight =
# PW x the real GT's
if [ -n "${PSEUDO:-}" ]; then
  bash $D/pseudo-sources.sh $PSEUDO $T/u0ok.json $(jq '[.sources[].weight // 1] | add' $X) > $T/ps.json
  jq -s '{sources: (.[0].sources + .[1].sources)}' $X $T/ps.json > $T/lab2.json; X=$T/lab2.json
fi
jq -s '{sources: (.[0].sources + .[1].sources)}' $X $T/u0ok.json > $OUT
echo "$(jq '[.sources[] | select(.ct_only != 1)] | length' $OUT) labelled + $(jq '[.sources[] | select(.ct_only == 1)] | length' $OUT) U0 sources -> $OUT"
rm -rf $T

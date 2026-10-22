#!/usr/bin/env bash
# scripts/universal/mksources.sh OUT.json: the universal run's sources (jq only): the labelled sources (labelled-desk.json: Paris 4 L2, 0139,
# 0814 -- winding codes + human ink-splat labels, holdout boxes) + every U0 scan prepped so far (scripts/u0/mksources.sh: ct_only, scanprep
# masks, the ~9.6 um level; PHerc1667, the test scroll, never). The U0 entries of the labelled scans (same frames) get the labelled
# holdouts; PHerc1447's community-surface eval box is held out too.
set -eu
OUT=${1:?out.json}; D=$(cd "$(dirname "$0")" && pwd); A=${A:-/vesuvius/ufsm/u0}
T=$(mktemp -d)
bash $A/scripts/mksources.sh $A/inventory.tsv $T/u0.json $T/u0-test.json > $T/log 2>&1 || { cat $T/log; exit 1; }
jq '.sources |= map(
      if (.ct | test("PHerc0139/20250728140407")) then . + {holdout: [[3472, 3472, 1424, 736, 736, 736]]}
      elif (.ct | test("PHerc0814/20250804134230")) then . + {holdout: [[2448, 2448, 2960, 736, 736, 736]]}
      elif (.ct | test("PHerc1447/20250521151220")) then . + {holdout: [[12280, 2820, 4890, 480, 480, 480]]}
      else . end)' $T/u0.json > $T/u0h.json
jq -s '{sources: (.[0].sources + .[1].sources)}' $D/labelled-desk.json $T/u0h.json > $OUT
echo "$(jq '[.sources[] | select(.ct_only != 1)] | length' $OUT) labelled + $(jq '[.sources[] | select(.ct_only == 1)] | length' $OUT) U0 sources -> $OUT"
rm -rf $T

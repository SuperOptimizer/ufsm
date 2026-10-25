#!/usr/bin/env bash
# addcnt.sh RES.tsv [RUNS-DIR]: add the turn-count columns (cnt_bias cnt_off cnt_offp dcnt_bias dcnt_off dcnt_offp, fit-compare's
# radial 0.5 mm segment counts by phase / density) to bench rows written before wbench.sh had them, from RUNS-DIR/TAG/BOX/cmp.log
# (default: the runs dir beside the res dir). Rewrites RES.tsv in place; a file that already has the columns is left alone.
set -euo pipefail; R=$1; D=${2:-$(dirname "$(dirname "$(readlink -f "$R")")")/runs}
head -1 "$R" | grep -q $'\tcnt_bias' && { echo "$R: has the columns"; exit 0; }
. <(sed -n '/^g_cnt()/,+2p' "$(dirname "$0")/wbench.sh")
T=$(mktemp "$R.XXXX")
{ IFS= read -r h; printf '%s\tcnt_bias\tcnt_off\tcnt_offp\tdcnt_bias\tdcnt_off\tdcnt_offp\n' "$h"
  while IFS= read -r l; do
    t=$(cut -f1 <<< "$l"); b=$(cut -f2 <<< "$l"); [ "$t" = tag ] && continue
    printf '%s\t%s\t%s\n' "$l" "$(g_cnt $D/$t/$b/cmp.log phase | tr ' ' '\t')" "$(g_cnt $D/$t/$b/cmp.log density | tr ' ' '\t')"
  done; } < "$R" > "$T"
mv "$T" "$R"; echo "$R: $(($(wc -l < "$R") - 1)) rows, $(awk -F'\t' 'NR > 1 && $NF == "na"' "$R" | wc -l) without counts"

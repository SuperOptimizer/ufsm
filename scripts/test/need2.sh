#!/usr/bin/env bash
# scripts/test/need2.sh [SHA_DIR]: close the 2-GPU gap gpuslot leaves (lead 2026-11-27). gpuslot's one-card slot cannot run gpu2 tests; a
# request whose only unrun tests were NEED2 leaves the queue with an "OPEN date ticket ref sha tests" line in ROOT/need2.txt. This runs the
# union of the OPEN lines' tests on a built master (SHA_DIR, default the newest built ROOT/<sha> of origin/master) on both cards (-g 0,1,
# run_suite's own gpuN.excl shared locks + slots: a trainer or an exclusive job holding a card makes them BUSY, never overlapping) and, if
# every one PASSes, marks those lines CLOSED. Run it when both cards are free (after a training run, in a swap gap).
# Exit: 0 all passed (lines closed), 1 a failure (lines stay OPEN, see the summary), 3 not run / BUSY, 4 nothing open.
set -u
ROOT=${SUITE_ROOT:-/vesuvius/ufsm/suite}; N=$ROOT/need2.txt
[ -s "$N" ] && grep -q '^OPEN ' "$N" || { echo "need2: nothing open in $N"; exit 4; }
D=${1:-}
if [ -z "$D" ]; then
  git -C "$(dirname "$0")/../.." fetch -q origin 2> /dev/null
  m=$(git -C "$(dirname "$0")/../.." rev-parse origin/master); D=$ROOT/${m:0:12}
fi
[ -x "$D/build-verify/test_lat" ] || { echo "need2: $D is not a built sha dir (gpuslot.sh add origin/master '^\$' + prep builds one)"; exit 3; }
tests=$(awk '$1 == "OPEN" {print $NF}' "$N" | tr ',' '\n' | sort -u | paste -sd'|')
rx="^(${tests})$"; O=$D/need2-$(date +%m%d%H%M); rm -rf -- "$O"
echo "need2: $(grep -c '^OPEN ' "$N") open lines, tests $tests, on $D"
( cd "$D" && tests/run_suite.sh -B build-verify -t all -g 0,1 -W 120 -o "$O" -K 'gpu2' "$rx" ); rc=$?
st=$(awk -F'\t' 'NR > 1 {c[$2]++} END {for (s in c) printf "%s %d ", s, c[s]}' "$O/results.tsv" 2> /dev/null)
echo "need2: rc $rc  $st ($O/summary.txt)"
if [ $rc = 0 ] && ! [[ "$st" =~ NEED2|BUSY|NOGPU|FAIL|TIMEOUT ]]; then
  sha=$(basename "$D")
  awk -v s="$sha" -v d="$(date '+%m-%d %H:%M')" '$1 == "OPEN" {$1 = "CLOSED"; $0 = $0 " (passed on " s " " d ")"} {print}' "$N" > "$N.tmp" && mv -f "$N.tmp" "$N"
  echo "need2: closed"; exit 0
fi
[ $rc = 1 ] && exit 1; exit 3

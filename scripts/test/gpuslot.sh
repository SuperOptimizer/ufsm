#!/usr/bin/env bash
# scripts/test/gpuslot.sh: the batched desk GPU slot (lead 2026-11-20). Agents queue the GPU tests their branch touches; the builds happen
# beforehand on the CPU; inksdk opens one short card slot (<= 15 min, daily or twice a day) and `drain` runs every queued request in it,
# back to back, so the card stops costing one window per agent.
#
#   gpuslot.sh add REF [REGEX]      queue REF (a commit of this clone) for the next slot: its touched fast-tier tests (run_suite -c against
#                                   its merge base with origin/master), or only the tests matching REGEX. Prints the ticket.
#   gpuslot.sh prep [-b N]          build every queued commit not built yet (ROOT/<sha>: release + verify, lowprio, -j N, default 4)
#   gpuslot.sh drain [-g GPU] [-m MIN] [-x S]
#                                   the slot: runs the built requests in queue order on GPU (default 1) until MIN minutes (default 15)
#                                   are used, each test at most S seconds (default 300; sanitizer tests ~racecheck / ~memcheck never run
#                                   here). Requests that did not fit stay queued. Run it while the slot's trainer has released the card.
#   gpuslot.sh status               the queue and the last results
# Results: ROOT/results.txt gets one line per request (time, ticket, ref, tests, PASS / FAIL counts, the summary path); the full runner
# output stays in ROOT/<sha>/slot-<ticket>/. ROOT: /vesuvius/ufsm/suite (desk_suite.sh's bare repo and sha dirs), env SUITE_ROOT.
# The GPU tests take the runner's normal locks (gpu<i>.excl shared + a slot): a trainer still holding the card exclusively makes them
# BUSY, they never preempt it.
set -u
ROOT=${SUITE_ROOT:-/vesuvius/ufsm/suite}; Q=$ROOT/slot.queue; RES=$ROOT/results.txt
mkdir -p "$ROOT" || exit 2
[ -d "$ROOT/repo.git" ] || git init -q --bare "$ROOT/repo.git" || exit 2
cmd=${1:-status}; shift
qlock() { exec 8> "$ROOT/slot.lock"; flock 8; }
built() { [ -f "$ROOT/${1:0:12}/.built" ]; }

case $cmd in
add)
  REF=${1:?REF}; RX=${2:-}
  [ "$REF" = HEAD ] && REF=$(git rev-parse --abbrev-ref HEAD)
  SHA=$(git rev-parse --verify "$REF^{commit}") || exit 2
  BASE=$(git merge-base "$SHA" origin/master 2> /dev/null || git merge-base "$SHA" master) || exit 2
  git push -q "$ROOT/repo.git" "$SHA:refs/suite/${SHA:0:12}" "$BASE:refs/suite/base-${BASE:0:12}" || exit 2
  qlock; T=$(date +%m%d%H%M%S)-$$
  printf '%s|%s|%s|%s|%s|%s\n' "$T" "$SHA" "$BASE" "$REF" "$RX" "${USER:-?}" >> "$Q"
  echo "gpuslot: queued $T: $REF = ${SHA:0:12} ($( [ -n "$RX" ] && echo "tests matching $RX" || echo "touched tests vs ${BASE:0:12}"))"
  ;;
prep)
  BJ=4; [ "${1:-}" = -b ] && BJ=$2
  [ -f "$Q" ] || { echo "gpuslot: queue empty"; exit 0; }
  for SHA in $(cut -d'|' -f2 "$Q" | awk '!s[$0]++'); do
    S=${SHA:0:12}; W=$ROOT/$S; built "$SHA" && continue
    echo "gpuslot: building $S"
    rm -rf -- "${ROOT:?}/$S"
    mkdir -p "$W" && git -C "$ROOT/repo.git" archive "$SHA" | tar -x -C "$W" || continue
    git -C "$ROOT/repo.git" log -1 --format=%H "$SHA" > "$W/.sha"
    ( cd "$W" && { cmake --preset release && cmake --preset verify; } > build.log 2>&1 &&
      nice -n 19 cmake --build build -j "$BJ" --target ufsm test_sheet make_pipeline_fixture test_http_reader test_sampler_safety test_formats >> build.log 2>&1 &&
      nice -n 19 cmake --build build-verify -j "$BJ" --target suite-bins >> build.log 2>&1 ) && touch "$W/.built" || echo "gpuslot: BUILD FAILED $W/build.log"
  done
  ;;
drain)
  G=1; MIN=15; X=300
  while getopts "g:m:x:" o; do case $o in g) G=$OPTARG;; m) MIN=$OPTARG;; x) X=$OPTARG;; *) exit 2;; esac; done
  [ -s "$Q" ] || { echo "gpuslot: queue empty"; exit 0; }
  qlock; END=$(( $(date +%s) + MIN * 60 )); : > "$Q.left"
  while IFS='|' read -r T SHA BASE REF RX WHO; do
    [ -z "$T" ] && continue
    S=${SHA:0:12}; W=$ROOT/$S; left=$(( END - $(date +%s) ))
    if ! built "$SHA" || [ $left -lt 60 ]; then echo "$T|$SHA|$BASE|$REF|$RX|$WHO" >> "$Q.left"; continue; fi
    O=$W/slot-$T; rm -rf -- "$O"
    # the request's tests: REGEX, or the touched ones (-c needs git: the sha dir is an archive, so its selection runs on a scratch clone)
    sel=(); if [ -n "$RX" ]; then sel=("$RX"); else
      git -C "$W" rev-parse -q --verify HEAD > /dev/null 2>&1 || { git -C "$W" init -q && git -C "$W" fetch -q "$ROOT/repo.git" "refs/suite/$S" "refs/suite/base-${BASE:0:12}" && git -C "$W" reset -q "$SHA"; }
      sel=(-c "$BASE"); fi
    echo "gpuslot: $T $REF ($S): ${left} s left in the slot"
    # (GPU tests only: the CPU half is the agent's own gate; never sanitizers on a shared card)
    ( cd "$W" && timeout -k 30 "$left" tests/run_suite.sh -B build-verify -t fast -g "$G" -x "$X" -W 60 -o "$O" -X '~(racecheck|memcheck)' \
        -K 'gpu|gpu2' "${sel[@]}" ) > "$O.log" 2>&1
    rc=$?
    sum=$(awk -F'\t' 'NR > 1 {c[$2]++} END {for (s in c) printf "%s %d ", s, c[s]}' "$O/results.tsv" 2> /dev/null)
    printf '%s %s %-28s rc %s  %s %s\n' "$(date '+%m-%d %H:%M')" "$T" "$REF" "$rc" "${sum:-no results}" "$O/summary.txt" | tee -a "$RES"
    [ $rc = 0 ] || [ $rc = 1 ] || echo "$T|$SHA|$BASE|$REF|$RX|$WHO" >> "$Q.left"   # (BUSY / timeout: stays queued)
  done < "$Q"
  mv -f "$Q.left" "$Q"
  ;;
status)
  echo "queue ($Q):"; [ -s "$Q" ] && while IFS='|' read -r T SHA BASE REF RX WHO; do
    printf '  %s %-28s %s %s %s\n' "$T" "$REF" "${SHA:0:12}" "$(built "$SHA" && echo built || echo "not built")" "${RX:-touched}"; done < "$Q"
  echo "last results ($RES):"; tail -n 8 "$RES" 2> /dev/null; true
  ;;
*) sed -n '2,20p' "$0"; exit 2;;
esac

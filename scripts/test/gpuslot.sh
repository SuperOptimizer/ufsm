#!/usr/bin/env bash
# scripts/test/gpuslot.sh: the batched desk GPU slot (lead 2026-11-20). Agents queue the GPU tests their branch touches; the builds happen
# beforehand on the CPU; inksdk opens one short card slot (<= 15 min, daily or twice a day) and `drain` runs every queued request in it,
# back to back, so the card stops costing one window per agent.
#
#   gpuslot.sh add REF [REGEX]      queue REF (a commit of this clone) for the next slot: its touched fast-tier tests (run_suite -c against
#                                   its merge base with origin/master), or only the tests matching REGEX. Prints the ticket.
#   gpuslot.sh prep [-b N]          build every queued commit not built yet (ROOT/<sha>: release + verify, lowprio, -j N, default 4)
#   gpuslot.sh drain [-g GPU] [-m MIN] [-x S] [-w MIN2]
#                                   the slot: first takes gpu<GPU>.excl SHARED, waiting up to MIN2 minutes (default 10) for a job that
#                                   holds it exclusively, and keeps it for the slot, so no exclusive job (an unroll, a trainer) starts on
#                                   the card meanwhile; then runs the built requests in queue order on GPU (default 1) until MIN minutes
#                                   (default 15) are used, each test at most S seconds (default 300; ~racecheck / ~memcheck never run here)
#   gpuslot.sh status               the queue and the last results
# Results: ROOT/results.txt gets one line per request (time, ticket, ref, rc, status counts, the summary path), and "SLOT BLOCKED by ..."
# when the card stayed held; the full runner output stays in ROOT/<sha>/slot-<ticket>/. ROOT: /vesuvius/ufsm/suite (desk_suite.sh's bare
# repo and sha dirs), env SUITE_ROOT. Requests stay queued while BUSY / not built / out of slot time; they leave the queue on a result
# (rc 0 / 1), on a selection error (rc 2), or after GPUSLOT_EXPIRE_H hours (default 72: an 11-21 request was retried for four days).
# Queue fields are separated by \x1f (11-25: a '|' separator split regexes like 'test_(a|b)'; a tab would collapse the empty REGEX
# field under read's whitespace IFS rules; old '|' lines are still read).
set -u
ROOT=${SUITE_ROOT:-/vesuvius/ufsm/suite}; Q=$ROOT/slot.queue; RES=$ROOT/results.txt; EXPH=${GPUSLOT_EXPIRE_H:-72}
LOCKD=${SUITE_LOCK_DIR:-/tmp/ufsm-suite-locks}
mkdir -p "$ROOT" || exit 2
[ -d "$ROOT/repo.git" ] || git init -q --bare "$ROOT/repo.git" || exit 2
cmd=${1:-status}; shift
qlock() { exec 8> "$ROOT/slot.lock"; flock 8; }
built() { [ -f "$ROOT/${1:0:12}/.built" ]; }
# one queue line -> T SHA BASE REF RX WHO (\x1f format, or the old T|SHA|BASE|REF|RX|WHO where RX may hold '|')
qparse() {
  local l=$1
  if [[ $l == *$'\x1f'* ]]; then IFS=$'\x1f' read -r T SHA BASE REF RX WHO <<< "$l"
  else IFS='|' read -r T SHA BASE REF _ <<< "$l"; WHO=${l##*|}; RX=${l#"$T|$SHA|$BASE|$REF|"}; RX=${RX%"|$WHO"}; fi
}
qline() { printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\n' "$T" "$SHA" "$BASE" "$REF" "$RX" "$WHO"; }
# age in hours of ticket T (mmddHHMMSS-pid, this year or the last)
tage() { local s=${1%%-*} y e; y=$(date +%Y); e=$(date -d "$y-${s:0:2}-${s:2:2} ${s:4:2}:${s:6:2}:${s:8:2}" +%s 2> /dev/null) || { echo 0; return; }
  [ "$e" -gt "$(date +%s)" ] && e=$(date -d "$((y - 1))-${s:0:2}-${s:2:2} ${s:4:2}:${s:6:2}:${s:8:2}" +%s); echo $(( ($(date +%s) - e) / 3600 )); }

case $cmd in
add)
  REF=${1:?REF}; RX=${2:-}
  [ "$REF" = HEAD ] && REF=$(git rev-parse --abbrev-ref HEAD)
  SHA=$(git rev-parse --verify "$REF^{commit}") || exit 2
  BASE=$(git merge-base "$SHA" origin/master 2> /dev/null || git merge-base "$SHA" master) || exit 2
  git push -q "$ROOT/repo.git" "$SHA:refs/suite/${SHA:0:12}" "$BASE:refs/suite/base-${BASE:0:12}" || exit 2
  qlock; T=$(date +%m%d%H%M%S)-$$; WHO=${USER:-?}
  qline >> "$Q"
  echo "gpuslot: queued $T: $REF = ${SHA:0:12} ($( [ -n "$RX" ] && echo "tests matching $RX" || echo "touched tests vs ${BASE:0:12}"))"
  ;;
prep)
  BJ=4; [ "${1:-}" = -b ] && BJ=$2
  [ -s "$Q" ] || { echo "gpuslot: queue empty"; exit 0; }
  declare -A seen
  while IFS= read -r l; do [ -z "$l" ] && continue; qparse "$l"
    S=${SHA:0:12}; W=$ROOT/$S; [ -n "${seen[$S]+x}" ] && continue; seen[$S]=1; built "$SHA" && continue
    [ -f "$W/build.failed" ] && continue   # (a failed build is not retried every prep)
    echo "gpuslot: building $S ($REF)"
    rm -rf -- "${ROOT:?}/$S"
    mkdir -p "$W" && git -C "$ROOT/repo.git" archive "$SHA" | tar -x -C "$W" || continue
    ( cd "$W" && { cmake --preset release && cmake --preset verify; } > build.log 2>&1 &&
      nice -n 19 cmake --build build -j "$BJ" --target ufsm test_sheet make_pipeline_fixture test_http_reader test_sampler_safety test_formats >> build.log 2>&1 &&
      nice -n 19 cmake --build build-verify -j "$BJ" --target suite-bins >> build.log 2>&1 ) && touch "$W/.built" ||
      { touch "$W/build.failed"; printf '%s %s %-28s BUILD FAILED %s\n' "$(date '+%m-%d %H:%M')" "$T" "$REF" "$W/build.log" | tee -a "$RES"; }
  done < "$Q"
  ;;
drain)
  G=1; MIN=15; X=300; WMIN=10
  while getopts "g:m:x:w:" o; do case $o in g) G=$OPTARG;; m) MIN=$OPTARG;; x) X=$OPTARG;; w) WMIN=$OPTARG;; *) exit 2;; esac; done
  [ -s "$Q" ] || { echo "gpuslot: queue empty"; exit 0; }
  qlock
  # reserve the card: a shared hold on gpu<G>.excl waits for an exclusive holder and keeps new exclusive jobs off it during the slot
  # (11-23 / 11-24: the slots found card 1 held exclusively by an unroll, so every request came back BUSY)
  mkdir -p "$LOCKD"; exec 7> "$LOCKD/gpu$G.excl"
  if ! flock -s -n 7; then
    h=$(lslocks -n -o PID,MODE,PATH 2> /dev/null | awk -v f="$LOCKD/gpu$G.excl" '$3 == f && $2 == "WRITE" {print $1; exit}')
    echo "gpuslot: card $G held by pid ${h:-?} $( [ -n "$h" ] && tr '\0' ' ' < /proc/$h/cmdline 2> /dev/null | cut -c1-100): waiting <= $WMIN min"
    if ! flock -s -w $((WMIN * 60)) 7; then
      printf '%s SLOT BLOCKED: card %s held exclusively by pid %s %s for %s min; %s requests stay queued\n' "$(date '+%m-%d %H:%M')" "$G" "${h:-?}" \
        "$( [ -n "$h" ] && tr '\0' ' ' < /proc/$h/cmdline 2> /dev/null | cut -c1-100)" "$WMIN" "$(grep -c . "$Q")" | tee -a "$RES"
      exit 3; fi
  fi
  END=$(( $(date +%s) + MIN * 60 )); : > "$Q.left"
  while IFS= read -r l; do
    [ -z "$l" ] && continue; qparse "$l"
    S=${SHA:0:12}; W=$ROOT/$S; left=$(( END - $(date +%s) )); age=$(tage "$T")
    if [ "$age" -ge "$EXPH" ]; then printf '%s %s %-28s EXPIRED after %s h in the queue\n' "$(date '+%m-%d %H:%M')" "$T" "$REF" "$age" | tee -a "$RES"; continue; fi
    [ -f "$W/build.failed" ] && { printf '%s %s %-28s dropped: build failed %s\n' "$(date '+%m-%d %H:%M')" "$T" "$REF" "$W/build.log" | tee -a "$RES"; continue; }
    if ! built "$SHA" || [ $left -lt 60 ]; then qline >> "$Q.left"; continue; fi
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
    [ -z "$sum" ] && sum="no results: $(grep -m1 -E 'run_suite:|invalid|error' "$O.log" | cut -c1-100)"
    printf '%s %s %-28s rc %s  %s %s\n' "$(date '+%m-%d %H:%M')" "$T" "$REF" "$rc" "$sum" "$O/summary.txt" | tee -a "$RES"
    case $rc in 0|1|2) ;; *) qline >> "$Q.left";; esac   # (BUSY / slot timeout: stays queued; a pass, a failure or a bad selection leave)
  done < "$Q"
  mv -f "$Q.left" "$Q"
  ;;
status)
  echo "queue ($Q):"; [ -s "$Q" ] && while IFS= read -r l; do [ -z "$l" ] && continue; qparse "$l"
    printf '  %s %-28s %s %-9s %4s h  %s\n' "$T" "$REF" "${SHA:0:12}" "$(built "$SHA" && echo built || { [ -f "$ROOT/${SHA:0:12}/build.failed" ] && echo FAILED || echo "not built"; })" "$(tage "$T")" "${RX:-touched}"; done < "$Q"
  echo "last results ($RES):"; tail -n 8 "$RES" 2> /dev/null; true
  ;;
*) sed -n '2,21p' "$0"; exit 2;;
esac

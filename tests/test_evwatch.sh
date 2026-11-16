#!/usr/bin/env bash
# scripts/rv/evwatch.sh with a stub eval.sh (CPU, no GPU, < 1 s): EVSC reaches eval.sh as ONE third argument (unset: an empty one, so
# eval.sh's default scroll list applies); a failing eval is retried and marked done after EVTRIES (2) tries with <step>.evalfail kept;
# a passing one is marked done. One loop pass each (<run>.evwatch.stop present: no sleep).
set -u
R=$(cd "$(dirname "$0")/.." && pwd); T=${TMPDIR:-/tmp}/test_evwatch.$$
case "$T" in */test_evwatch.*) ;; *) echo "FAIL: odd temp dir $T"; exit 1;; esac
trap 'rm -rf -- "$T"' EXIT
mkdir -p "$T/s" "$T/r/snap"; cp "$R/scripts/rv/evwatch.sh" "$T/s/"
fail() { echo "FAIL: $*"; exit 1; }
printf '#!/bin/bash\necho "ARGS $# [${3-}]"\n' > "$T/s/eval.sh"; touch "$T/r/snap/5000.ckpt" "$T/r/snap/2500.ckpt" "$T/r.evwatch.stop"
PAUSE=0 EVGPU=1 bash "$T/s/evwatch.sh" "$T/r" || fail "evwatch run 1"
grep -q '^ARGS 3 \[\]$' "$T/r.evwatch.log" || fail "EVSC unset: eval.sh did not get an empty third argument ($(grep ARGS "$T/r.evwatch.log"))"
[ -e "$T/r/snap/5000.evaldone" ] && [ ! -e "$T/r/snap/2500.evaldone" ] || fail "evaldone marks (step % EVERY)"
rm "$T/r/snap/5000.evaldone"
EVSC="p1667 p139" PAUSE=0 EVGPU=1 bash "$T/s/evwatch.sh" "$T/r" || fail "evwatch run 2"
[ "$(grep ARGS "$T/r.evwatch.log" | tail -1)" = "ARGS 3 [p1667 p139]" ] || fail "EVSC=\"p1667 p139\": $(grep ARGS "$T/r.evwatch.log" | tail -1)"
printf '#!/bin/bash\nexit 3\n' > "$T/s/eval.sh"; rm "$T/r/snap/5000.evaldone"
PAUSE=0 EVGPU=1 bash "$T/s/evwatch.sh" "$T/r"; [ ! -e "$T/r/snap/5000.evaldone" ] || fail "a failed eval was marked done on its first try"
PAUSE=0 EVGPU=1 bash "$T/s/evwatch.sh" "$T/r"; [ -e "$T/r/snap/5000.evaldone" ] && [ "$(cat "$T/r/snap/5000.evalfail")" = 2 ] || fail "after 2 failed tries: done + evalfail 2"
[ "$(grep -c 'FAILED (rc 3' "$T/r.evwatch.log")" = 2 ] || fail "failure log lines"
echo "test_evwatch: ok"

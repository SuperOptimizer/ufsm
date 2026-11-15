#!/usr/bin/env bash
# test_suite_kill.sh: run_suite.sh kills what a failed test left running. A one-test manifest whose test starts a child that ignores
# SIGTERM (as ufsm train does: it checkpoints on it) and exits 1; afterwards no process may carry that run's UFSM_SUITE_RUN marker.
# Before (2026-11-15) such a child outlived the test: timeout -k escalates only while its own child lives.
set -u
H=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d "${TMPDIR:-/tmp}/suitekill.XXXXXX"); case "$W" in */suitekill.*) ;; *) echo "FAIL: mktemp"; exit 1;; esac
trap 'rm -rf -- "$W"' EXIT
TOK="SK_TOKEN=$W"   # (the child's own mark, independent of the runner's: a runner without the marker leaves it findable too)
cat > "$W/suite.txt" << M
sk_leaver                                      | cpu          |     1 | bash -c 'env $TOK bash -c "trap \\"\\" TERM; exec sleep 300" & sleep 0.5; exit 1'
M
SUITE_MANIFEST="$W/suite.txt" "$H/tests/run_suite.sh" -t all -g none -j 1 -o "$W/out" sk_leaver > "$W/run.log" 2>&1
grep -q "FAIL" "$W/run.log" || { echo "FAIL: the leaver test did not fail as written"; cat "$W/run.log"; exit 1; }
left=$(grep -lzsx "$TOK" /proc/[0-9]*/environ 2> /dev/null | sed -E 's|/proc/([0-9]+)/environ|\1|' | tr '\n' ' ')
[ -z "$left" ] && grep -q "killed what the test left running" "$W/out/logs/sk_leaver.log"* 2> /dev/null && { echo "test_suite_kill: ok"; exit 0; }
echo "FAIL: processes left by the failed test: ${left:-none, but no kill line in the log}"; [ -n "$left" ] && kill -KILL $left 2> /dev/null; exit 1

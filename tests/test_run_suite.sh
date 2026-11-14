#!/usr/bin/env bash
# test_run_suite: the runner's own contract (2026-11-14, ops audit #2 / #7), on a mock manifest; touches no GPU.
#   exit 0 all pass | 1 a failure | 3 GPU tests not run (NOGPU / BUSY / NEED2) | -I makes 3 into 0;
#   a card whose gpu<i>.excl is held exclusively (a training run) -> BUSY after -W s, with the holder's pid in the test log.
set -u
H=$(cd "$(dirname "$0")/.." && pwd); D=${TMPDIR:-/tmp}/trs.$$; mkdir -p "$D"; trap '[ -n "$LP" ] && { pkill -P "$LP"; kill "$LP"; } 2> /dev/null; rm -rf -- "$D"' EXIT; LP=
M=$D/manifest.txt; R=$H/tests/run_suite.sh; fails=0
cat > "$M" <<'EOF'
t_pass  |       | 1 | true
t_skip  |       | 1 | echo SKIP nothing to do
t_gpu   | gpu   | 1 | true
t_gpu2  | gpu2  | 1 | true
t_fail  | slow  | 1 | false
EOF
run() { SUITE_MANIFEST=$M SUITE_LOCK_DIR=$D/locks timeout 120 "$R" -o "$D/out.$1" "${@:2}" > "$D/log.$1" 2>&1; echo $?; }
st() { awk -F'\t' -v t="$2" '$1 == t {print $2}' "$D/out.$1/results.tsv"; }
check() { [ "$2" = "$3" ] && echo "ok: $1" || { echo "FAIL: $1: got '$2', want '$3'"; fails=$((fails + 1)); }; }

check "CPU tests only, -g none: exit 3 (INCOMPLETE)" "$(run a -g none 't_pass|t_skip|^t_gpu$')" 3
check "-g none: t_gpu NOGPU" "$(st a t_gpu)" NOGPU
check "t_skip reported SKIP" "$(st a t_skip)" SKIP
grep -q 'INCOMPLETE: 1 GPU tests not run' "$D/log.a" && echo "ok: summary says INCOMPLETE" || { echo "FAIL: no INCOMPLETE line"; fails=$((fails + 1)); }
check "-I: exit 0" "$(run b -g none -I 't_pass|^t_gpu$')" 0
check "CPU only selection: exit 0" "$(run c -g none 't_pass|t_skip')" 0
check "a failure: exit 1 (beats INCOMPLETE)" "$(run d -g none -t all 't_fail|^t_gpu$')" 1
check "one GPU listed: an untagged gpu2 test is NEED2" "$(run e -g 0 -F -W 2 't_gpu2'; st e t_gpu2)" "3
NEED2"
# a training run's exclusive lock on GPU 0: the test waits -W s and is BUSY, the log names the holder (no GPU is touched: never started)
mkdir -p "$D/locks"; flock -x "$D/locks/gpu0.excl" sleep 300 > /dev/null 2>&1 < /dev/null & LP=$!; sleep 0.5
check "card held exclusively: exit 3" "$(run f -g 0 -F -W 2 '^t_gpu$')" 3
check "card held exclusively: BUSY" "$(st f t_gpu)" BUSY
grep -q 'flock -x' "$D/out.f/logs/t_gpu.log" && echo "ok: the holder is named" || { echo "FAIL: no holder in the log"; cat "$D/out.f/logs/t_gpu.log"; fails=$((fails + 1)); }
[ $fails = 0 ] && echo "test_run_suite: ok" || { echo "test_run_suite: $fails failure(s)"; exit 1; }

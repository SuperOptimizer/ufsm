#!/usr/bin/env bash
# tests/test_gpuslot_prep.sh: scripts/test/gpuslot.sh's builders leave other builders' sha dirs alone (lead 11-27: three dirs wiped mid-build
# that day) and a failed build keeps its ticket. On a scratch SUITE_ROOT with a stand-in build command (GPUSLOT_BUILD_CMD), no cmake, no GPU:
#  1. a .prebuilt dir (a hand build) is skipped and left byte-for-byte;  2. a sha whose lock another process holds is skipped and untouched;
#  3. a lock-less dir with a fresh build.log is skipped;  4. a plain request builds (.built);  5. a failing build writes build.failed and one
#  BUILD FAILED line, and drain keeps the ticket queued with one "kept queued" note (not "dropped").  CPU only, ~2 s.
set -u
H=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); G=$H/scripts/test/gpuslot.sh
D=$(mktemp -d "${TMPDIR:-/tmp}/test_gpuslot_prep_XXXXXX"); R=$D/suite
fail() { echo "FAIL: $*"; echo "kept $D"; exit 1; }
export SUITE_ROOT=$R SUITE_LOCK_DIR=$D/locks GPUSLOT_BUILD_CMD='[ -e FAILME ] && exit 1; touch built-by-stand-in'
git init -q "$D/src" && cd "$D/src" || fail "git init"
git config user.email t@t && git config user.name t
mk() { echo "$1" > f; [ "${2:-}" = fail ] && touch FAILME || rm -f FAILME; git add -A > /dev/null; git commit -qm "$1"; git rev-parse HEAD; }
base=$(mk base); git branch -q -M master; git update-ref refs/remotes/origin/master "$base"
A=$(mk a); B=$(mk b); C=$(mk c); E=$(mk e); F=$(mk f fail)
for s in $A $B $C $E $F; do bash "$G" add "$s" > /dev/null || fail "add $s"; done
a=${A:0:12}; b=${B:0:12}; c=${C:0:12}; e=${E:0:12}; f=${F:0:12}
mkdir -p "$R/$a" && echo hand > "$R/$a/handfile" && touch "$R/$a/.prebuilt"                       # 1: prebuilt by hand
mkdir -p "$R/.locks" "$R/$b" && echo partial > "$R/$b/partial"                                        # 2: another builder's lock
( exec 9> "$R/.locks/$b.lock"; flock 9; sleep 20 ) & hold=$!; sleep 0.5
mkdir -p "$R/$c" && echo log > "$R/$c/build.log"                                                      # 3: fresh build.log, no lock
out=$(bash "$G" prep 2>&1)
kill $hold 2> /dev/null; wait $hold 2> /dev/null
[ "$(cat "$R/$a/handfile" 2> /dev/null)" = hand ] && [ ! -f "$R/$a/.built" ] || fail "prebuilt dir touched: $out"
grep -q "$a .*prebuilt" <<< "$out" || fail "no prebuilt skip message: $out"
[ "$(cat "$R/$b/partial" 2> /dev/null)" = partial ] || fail "a locked sha dir was touched: $out"
grep -q "$b .*another builder" <<< "$out" || fail "no lock skip message: $out"
[ -f "$R/$c/build.log" ] && [ ! -f "$R/$c/.built" ] || fail "a fresh build.log dir was rebuilt: $out"
[ -f "$R/$e/.built" ] && [ -f "$R/$e/built-by-stand-in" ] || fail "a plain request was not built: $out"
[ -f "$R/$f/build.failed" ] && [ "$(grep -c "BUILD FAILED" "$R/results.txt")" = 1 ] || fail "a failing build left no build.failed / one BUILD FAILED line"
bash "$G" build "$B" > /dev/null 2>&1; [ -f "$R/$b/.built" ] || fail "build SHA did not build once the lock was free"
# drain: the failed request stays queued with one note (twice: the note is not repeated); no GPU tests run (nothing selected is built here)
for i in 1 2; do bash "$G" drain -g 7 -m 1 -w 0 > /dev/null 2>&1; done
grep -q "$F" "$R/slot.queue" || fail "drain dropped the failed-build ticket"
[ "$(grep -c "kept queued" "$R/results.txt")" = 1 ] || fail "want one 'kept queued' note, got $(grep -c 'kept queued' "$R/results.txt")"
! grep -q "dropped: build failed" "$R/results.txt" || fail "drain still drops failed builds"
rm -rf -- "$D"; echo "test_gpuslot_prep: ok"

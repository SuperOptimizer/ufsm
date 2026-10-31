#!/usr/bin/env bash
# scripts/test/desk_suite.sh: run the ufsm suite for a branch / commit on the desk (2x RTX 5060 Ti, 32 threads), from the laptop or
# any clone. The commit is pushed over ssh into the desk's bare repo /vesuvius/ufsm/suite/repo.git (nothing has to be on GitHub),
# exported to /vesuvius/ufsm/suite/<sha>, built (ccache, shared across the sha dirs) and tested at nice 10.
#
#   scripts/test/desk_suite.sh [-t fast|slow|all] [-j N] [-T N] [-m GB] [-l LOAD] [-w MIN] [-b N] [-k] REF [REGEX...]
#     REF     branch, tag or commit of the local repo (uncommitted changes are not sent)
#     -t      tier (default fast = make test); -j / -T: runner jobs / threads per test (default 8 / 3)
#     -m GB -l LOAD: before the build and before the tests the desk must have MemAvailable >= GB (default 40) and a 1-min
#             load average <= LOAD (default 32 = the thread count); -w MIN: wait that long for it (default 60), then give up (exit 3)
#     -b N    build jobs (default 12); -k keeps the sha dir after a pass (failed runs are always kept)
#     REGEX   only tests whose name matches (runner filter)
#     env SUITE_ARGS: more runner options (e.g. "-P" profile, "-x 600")
# Only one desk suite runs at a time (flock /vesuvius/ufsm/suite/suite.lock; later callers wait and say so). GPUs: those with
# >= 3 GB free, most free first (the runner's GPU locks keep it to one suite test per GPU); none free -> GPU tests are SKIPped.
# Output: build log tail on a build failure, the runner's progress, failing tests' logs and the summary. Logs stay on the desk in
# /vesuvius/ufsm/suite/<sha>/out (the last 8 sha dirs are kept). Exit = the runner's (0 = all passed).
set -u
DESK=${DESK:-forrest@192.168.0.175}; ROOT=/vesuvius/ufsm/suite
TIER=fast; J=8; T=3; MEMGB=40; LOADMAX=32; WAITMIN=60; BJ=12; KEEP=0
while getopts "t:j:T:m:l:w:b:kh" o; do case $o in
  t) TIER=$OPTARG;; j) J=$OPTARG;; T) T=$OPTARG;; m) MEMGB=$OPTARG;; l) LOADMAX=$OPTARG;; w) WAITMIN=$OPTARG;; b) BJ=$OPTARG;; k) KEEP=1;; *) sed -n '2,19p' "$0"; exit 2;; esac; done
shift $((OPTIND - 1)); [ $# -ge 1 ] || { sed -n '2,19p' "$0"; exit 2; }
REF=$1; shift
SHA=$(git rev-parse --verify "$REF^{commit}") || exit 2
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || exit 2
FILT=""; for f in "$@"; do FILT+=" $(printf '%q' "$f")"; done
echo "desk_suite: $REF = ${SHA:0:12} ($(git log -1 --format=%s "$SHA" | cut -c1-80))"

ssh -o ConnectTimeout=10 "$DESK" "mkdir -p $ROOT && { [ -d $ROOT/repo.git ] || git init -q --bare $ROOT/repo.git; }" || exit 2
git push -q "$DESK:$ROOT/repo.git" "$SHA:refs/suite/${SHA:0:12}" 2>&1 | grep -v '^$' || true

# the desk side as a script file, run on a pty (ssh -tt: a dropped caller HUPs it); its arguments quoted (ssh joins the words into one
# string: quote each so empty ones survive)
RS=$ROOT/.remote-${SHA:0:12}.sh
ssh -o ConnectTimeout=10 "$DESK" "cat > $RS" <<'REMOTE' || exit 2

set -u
SHA=$1 TIER=$2 J=$3 T=$4 MEMGB=$5 BJ=$6 KEEP=$7 FILT=$8 XARGS=$9 LOADMAX=${10} WAITMIN=${11}
ROOT=/vesuvius/ufsm/suite; S=${SHA:0:12}; W=$ROOT/$S
export PATH=/usr/local/cuda/bin:$PATH CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
deskok() {   # wait (<= WAITMIN) until MemAvailable >= MEMGB and the 1-min load <= LOADMAX: the desk's own jobs come first
  local i a l n=$(( WAITMIN * 2 ))
  for ((i = 0; i <= n; i++)); do
    a=$(awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); l=$(cut -d' ' -f1 /proc/loadavg)
    [ "$a" -ge "$MEMGB" ] && awk -v l="$l" -v m="$LOADMAX" 'BEGIN {exit !(l <= m)}' && return 0
    [ $i = 0 ] && echo "desk_suite: desk busy (MemAvailable ${a} GB, need ${MEMGB}; load ${l}, max ${LOADMAX}): waiting <= ${WAITMIN} min"
    [ $i -lt $n ] && sleep 30; done
  echo "desk_suite: desk still busy (MemAvailable ${a} GB, load ${l}): giving up, try later"; return 1; }
# the caller gone (ssh dropped, ^C on the laptop): no orphan build or suite on the desk. The script runs on a pty (ssh -tt), so the
# desk side gets SIGHUP; the build and the runner lead their own process groups (setsid) and get a TERM (the runner's trap stops its
# tests). (Without a pty sshd keeps the session open while the child lives: nothing tells a sleeping script that the caller left.)
stty -onlcr 2> /dev/null   # (no CR LF on the pty)
GRP=$ROOT/.$S.pgid; : > $GRP
gone() { g=$(cat $GRP 2> /dev/null); [ -n "$g" ] && kill -TERM -- -"$g" 2> /dev/null; rm -f $GRP; exit 129; }
trap gone HUP INT TERM; trap 'rm -f $GRP "$0"' EXIT
inbg() { setsid "$@" < /dev/null & local p=$!; echo $p > $GRP; wait $p; local r=$?; : > $GRP; return $r; }
exec 9> $ROOT/suite.lock
flock -n 9 || { echo "desk_suite: another desk suite holds $ROOT/suite.lock: waiting"; flock 9; }
deskok || exit 3
if [ ! -f $W/.built ]; then
  [ -d $W ] && rm -rf -- "$ROOT/$S"
  mkdir -p $W && git -C $ROOT/repo.git archive "$SHA" | tar -x -C $W || exit 2
  echo "desk_suite: building in $W (make -j $BJ, nice 10)"; t0=$(date +%s)
  cd $W || exit 2
  if ! inbg bash -c "nice -n 10 make -j $BJ VERIFY=0 build/ufsm build/test_sheet build/make_pipeline_fixture build/test_http_reader \
          build/test_sampler_safety build/test_formats > build.log 2>&1 && nice -n 10 make -j $BJ VERIFY=1 suite-bins >> build.log 2>&1"; then
    echo "desk_suite: BUILD FAILED ($W/build.log):"; grep -E 'error|Error' $W/build.log | head -30; tail -n 15 $W/build.log; exit 2; fi
  touch $W/.built; echo "desk_suite: built in $(( $(date +%s) - t0 )) s"
fi
deskok || exit 3
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 3072' | sort -t, -k2,2nr | cut -d, -f1 | paste -sd, -)
[ -z "$G" ] && { echo "desk_suite: no GPU with 3 GB free: GPU tests are skipped"; G=none; }
rm -rf -- "$W/out"
cd $W && eval inbg nice -n 10 tests/run_suite.sh -t "$TIER" -j "$J" -T "$T" -g "$G" -o "$W/out" -B build-verify $XARGS $FILT; rc=$?
# keep the last 8 sha dirs (a passed one only with -k)
[ $rc = 0 ] && [ "$KEEP" = 0 ] && rm -rf -- "$ROOT/$S/build" "$ROOT/$S/build-verify"
ls -dt $ROOT/[0-9a-f]*/ 2> /dev/null | tail -n +9 | while read -r d; do d=${d%/}; [[ ${d##*/} =~ ^[0-9a-f]{12}$ ]] && rm -rf -- "$ROOT/${d##*/}"; done
exit $rc
REMOTE
ssh -tt -o ConnectTimeout=10 "$DESK" "bash $RS $(printf '%q ' "$SHA" "$TIER" "$J" "$T" "$MEMGB" "$BJ" "$KEEP" "$FILT" "${SUITE_ARGS:-}" "$LOADMAX" "$WAITMIN")"

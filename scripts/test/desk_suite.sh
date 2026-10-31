#!/usr/bin/env bash
# scripts/test/desk_suite.sh: run the ufsm suite for a branch / commit on the desk (2x RTX 5060 Ti, 32 threads), from the laptop or
# any clone. The commit is pushed over ssh into the desk's bare repo /vesuvius/ufsm/suite/repo.git (nothing has to be on GitHub),
# exported to /vesuvius/ufsm/suite/<sha>, built (ccache, shared across the sha dirs) and tested at nice 10.
#
#   scripts/test/desk_suite.sh [-t fast|slow|all] [-j N] [-T N] [-m GB] [-b N] [-k] REF [REGEX...]
#     REF     branch, tag or commit of the local repo (uncommitted changes are not sent)
#     -t      tier (default fast = make test); -j / -T: runner jobs / threads per test (default 8 / 3)
#     -m GB   the desk needs MemAvailable >= GB before building and testing (default 40; waits up to 30 min, then gives up)
#     -b N    build jobs (default 12); -k keeps the sha dir after a pass (failed runs are always kept)
#     REGEX   only tests whose name matches (runner filter)
#     env SUITE_ARGS: more runner options (e.g. "-P" profile, "-x 600")
# Only one desk suite runs at a time (flock /vesuvius/ufsm/suite/suite.lock; later callers wait and say so). GPUs: those with
# >= 3 GB free, most free first (the runner's GPU locks keep it to one suite test per GPU); none free -> GPU tests are SKIPped.
# Output: build log tail on a build failure, the runner's progress, failing tests' logs and the summary. Logs stay on the desk in
# /vesuvius/ufsm/suite/<sha>/out (the last 8 sha dirs are kept). Exit = the runner's (0 = all passed).
set -u
DESK=${DESK:-forrest@192.168.0.175}; ROOT=/vesuvius/ufsm/suite
TIER=fast; J=8; T=3; MEMGB=40; BJ=12; KEEP=0
while getopts "t:j:T:m:b:kh" o; do case $o in
  t) TIER=$OPTARG;; j) J=$OPTARG;; T) T=$OPTARG;; m) MEMGB=$OPTARG;; b) BJ=$OPTARG;; k) KEEP=1;; *) sed -n '2,18p' "$0"; exit 2;; esac; done
shift $((OPTIND - 1)); [ $# -ge 1 ] || { sed -n '2,18p' "$0"; exit 2; }
REF=$1; shift
SHA=$(git rev-parse --verify "$REF^{commit}") || exit 2
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || exit 2
FILT=""; for f in "$@"; do FILT+=" $(printf '%q' "$f")"; done
echo "desk_suite: $REF = ${SHA:0:12} ($(git log -1 --format=%s "$SHA" | cut -c1-80))"

ssh -o ConnectTimeout=10 "$DESK" "mkdir -p $ROOT && { [ -d $ROOT/repo.git ] || git init -q --bare $ROOT/repo.git; }" || exit 2
git push -q "$DESK:$ROOT/repo.git" "$SHA:refs/suite/${SHA:0:12}" 2>&1 | grep -v '^$' || true

ssh -o ConnectTimeout=10 "$DESK" bash -s -- "$SHA" "$TIER" "$J" "$T" "$MEMGB" "$BJ" "$KEEP" "$FILT" "${SUITE_ARGS:-}" <<'REMOTE'
set -u
SHA=$1 TIER=$2 J=$3 T=$4 MEMGB=$5 BJ=$6 KEEP=$7 FILT=$8 XARGS=$9
ROOT=/vesuvius/ufsm/suite; S=${SHA:0:12}; W=$ROOT/$S
export PATH=/usr/local/cuda/bin:$PATH CCACHE_BASEDIR=$W CCACHE_NOHASHDIR=1
memok() {   # wait (<= 30 min) for MemAvailable >= MEMGB
  local i a; for ((i = 0; i < 60; i++)); do
    a=$(awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ "$a" -ge "$MEMGB" ] && return 0
    [ $i = 0 ] && echo "desk_suite: MemAvailable ${a} GB < ${MEMGB} GB, waiting (<= 30 min)"; sleep 30; done
  echo "desk_suite: MemAvailable still below ${MEMGB} GB: giving up (desk jobs come first)"; return 1; }
exec 9> $ROOT/suite.lock
flock -n 9 || { echo "desk_suite: another desk suite holds $ROOT/suite.lock: waiting"; flock 9; }
memok || exit 3
if [ ! -f $W/.built ]; then
  [ -d $W ] && rm -rf -- "$ROOT/$S"
  mkdir -p $W && git -C $ROOT/repo.git archive "$SHA" | tar -x -C $W || exit 2
  echo "desk_suite: building in $W (make -j $BJ, nice 10)"; t0=$(date +%s)
  if ! (cd $W && nice -n 10 make -j "$BJ" VERIFY=0 build/ufsm build/test_sheet build/make_pipeline_fixture build/test_http_reader \
          build/test_sampler_safety build/test_formats > build.log 2>&1 && nice -n 10 make -j "$BJ" VERIFY=1 suite-bins >> build.log 2>&1); then
    echo "desk_suite: BUILD FAILED ($W/build.log):"; grep -E 'error|Error' $W/build.log | head -30; tail -n 15 $W/build.log; exit 2; fi
  touch $W/.built; echo "desk_suite: built in $(( $(date +%s) - t0 )) s"
fi
memok || exit 3
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 3072' | sort -t, -k2,2nr | cut -d, -f1 | paste -sd, -)
[ -z "$G" ] && { echo "desk_suite: no GPU with 3 GB free: GPU tests are skipped"; G=none; }
rm -rf -- "$W/out"
cd $W && eval nice -n 10 tests/run_suite.sh -t "$TIER" -j "$J" -T "$T" -g "$G" -o "$W/out" -B build-verify $XARGS $FILT; rc=$?
# keep the last 8 sha dirs (a passed one only with -k)
[ $rc = 0 ] && [ "$KEEP" = 0 ] && rm -rf -- "$ROOT/$S/build" "$ROOT/$S/build-verify"
ls -dt $ROOT/[0-9a-f]*/ 2> /dev/null | tail -n +9 | while read -r d; do d=${d%/}; [[ ${d##*/} =~ ^[0-9a-f]{12}$ ]] && rm -rf -- "$ROOT/${d##*/}"; done
exit $rc
REMOTE

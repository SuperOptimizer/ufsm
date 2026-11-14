#!/usr/bin/env bash
# scripts/test/pod_gate.sh: the GPU half of the merge gate on the perf pod (2026-11-14, ops audit #2). The desk's GPUs belong to the
# training run, so a desk suite runs the CPU tests only (run_suite.sh reports the GPU ones NOGPU / BUSY and exits 3 = INCOMPLETE).
# This runs the tests REF's changes can affect (run_suite.sh -c BASE) on the pod's GPU, built from REF.
#
#   scripts/test/pod_gate.sh [-b BASE] [-t TIER] [-j N] [-x S] [-a] REF [REGEX...]
#     REF    branch / commit to test (uncommitted changes are not sent)       BASE  default origin/master (the merge base with REF)
#     -t     tier (default fast)   -j  build jobs (default 6: the pod's cgroup has ~10 CPUs and A/B runs share it)
#     -x     the whole run's timeout in seconds (default 3600)   -a  every test of the tier (no -c selection)
#     env POD (root@213.173.109.173), POD_PORT (20326), SUITE_ARGS (more runner options)
# The pod side: bare repo /root/gate/repo.git (REF and BASE pushed as refs/gate/*), a clone per sha in /root/gate/<sha12> (git, so the
# runner's -c can diff), the build under flock /root/build.lock, the tests under flock /root/gpu.lock (the pod convention: GPU A/B
# timing runs hold it, so a gate never builds or tests beside one: the build waits for the GPU lock too) and timeout -k inside it.
# The last 6 sha dirs are kept. Exit = the runner's: 0 passed, 1 failures, 3 incomplete; 2 build / setup errors; 4 lock timeout.
set -u
POD=${POD:-root@213.173.109.173}; PORT=${POD_PORT:-20326}; BASE=origin/master; TIER=fast; BJ=6; TO=3600; ALL=0
while getopts "b:t:j:x:ah" o; do case $o in
  b) BASE=$OPTARG;; t) TIER=$OPTARG;; j) BJ=$OPTARG;; x) TO=$OPTARG;; a) ALL=1;; *) sed -n '2,16p' "$0"; exit 2;; esac; done
shift $((OPTIND - 1)); [ $# -ge 1 ] || { sed -n '2,16p' "$0"; exit 2; }
REF=$1; shift
SHA=$(git rev-parse --verify "$REF^{commit}") || exit 2
MB=$(git merge-base "$SHA" "$(git rev-parse --verify "$BASE^{commit}")") || exit 2
[[ $SHA =~ ^[0-9a-f]{40}$ && $MB =~ ^[0-9a-f]{40}$ ]] || exit 2
FILT=""; for f in "$@"; do FILT+=" $(printf '%q' "$f")"; done
SSH=(ssh -o ConnectTimeout=15 -o BatchMode=yes -p "$PORT" "$POD")
echo "pod_gate: $REF = ${SHA:0:12} ($(git log -1 --format=%s "$SHA" | cut -c1-80)), base ${MB:0:12}, tier $TIER"
"${SSH[@]}" 'mkdir -p /root/gate && { [ -d /root/gate/repo.git ] || git init -q --bare /root/gate/repo.git; }' || exit 2
git push -q -f "ssh://$POD:$PORT/root/gate/repo.git" "$SHA:refs/gate/${SHA:0:12}" "$MB:refs/gate/base-${MB:0:12}" || exit 2
SEL="-c $MB"; [ $ALL = 1 ] && SEL=
"${SSH[@]}" "bash -s -- $(printf '%q ' "$SHA" "$TIER" "$BJ" "$TO" "$SEL" "${SUITE_ARGS:-}" "$FILT")" <<'REMOTE'
set -u
SHA=$1 TIER=$2 BJ=$3 TO=$4 SEL=$5 XARGS=$6 FILT=$7
G=/root/gate; S=${SHA:0:12}; W=$G/$S
if [ ! -f $W/.built ]; then
  case "$W" in /root/gate/[0-9a-f]*) rm -rf -- "$W";; esac
  git clone -q --shared $G/repo.git $W && git -C $W -c advice.detachedHead=false checkout -q "$SHA" || exit 2
  echo "pod_gate: waiting for /root/build.lock and /root/gpu.lock (no build beside an A/B timing run)"
  cd $W || exit 2
  # (the GPU lock too: the A/B timing runs hold it, and a build next to one distorts it; CCACHE_BASEDIR: hits across the sha dirs)
  if ! timeout -k 30 "$TO" flock -w "$TO" /root/build.lock flock -w "$TO" /root/gpu.lock bash -c "
        { cmake --preset release && cmake --preset verify; } > build.log 2>&1 &&
        nice -n 5 cmake --build build -j $BJ --target ufsm test_sheet make_pipeline_fixture test_http_reader test_sampler_safety test_formats >> build.log 2>&1 &&
        nice -n 5 cmake --build build-verify -j $BJ --target suite-bins >> build.log 2>&1"; then
    echo "pod_gate: BUILD FAILED or lock timeout ($W/build.log):"; grep -E 'error|Error' $W/build.log | head -30; tail -n 15 $W/build.log; exit 2; fi
  touch $W/.built
fi
cd $W || exit 2
git fetch -q $G/repo.git 'refs/gate/*:refs/gate/*' 2> /dev/null
rm -rf -- "$W/out"
eval timeout -k 30 "$TO" flock -w "$TO" /root/gpu.lock timeout -k 30 "$TO" tests/run_suite.sh -t "$TIER" $SEL -o "$W/out" -B build-verify $XARGS $FILT; rc=$?
[ $rc = 124 ] && { echo "pod_gate: timed out (lock or run) after $TO s"; rc=4; }
ls -dt $G/[0-9a-f]*/ 2> /dev/null | tail -n +7 | while read -r d; do d=${d%/}; [[ ${d##*/} =~ ^[0-9a-f]{12}$ ]] && rm -rf -- "$G/${d##*/}"; done
exit $rc
REMOTE

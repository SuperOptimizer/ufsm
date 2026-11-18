#!/usr/bin/env bash
# scripts/pod/podq3.sh: the pod queue after podq2's unroll (lead 11-18: "use the pod for whatever we need"), one GPU job at a time under
# /root/gpu.lock, each under timeout -k, chained here (no waiting on messages); the desk pulls the results (scripts/pod/q2pull.sh).
#   smoke  inksdk's ink smoke (kit/podq.sh): waits for a running one (started 16:22 pod time), else runs it; passed = K/state/ALLDONE
#          with "ALLDONE: smoke" in K/podq.log
#   gate   the GPU merge gate of master COMMIT on sm_120 (pod_gate.sh's recipe on the pod: clone COMMIT, release + verify builds, the fast
#          tier, every test: -a) -> /root/gate/<sha12>/out (the report: out/summary.txt + logs)
#   full   only if the smoke passed: inksdk's FULL=1 ink retrain (kit/podq.sh from kit/podq.sh.new, write-new + mv between the smoke and
#          FULL; COMMIT >= 9bfa0298), its own gates; timeout FULLTO (20 h)
# State: Q3/state/<step>.done; Q3/state/ALLDONE at the end (the desk pull's end marker; the pod is then idle). Stop after a step: touch Q3/STOPQ.
# Env: COMMIT (required), K (/root/ink), Q3 (/root/pod/q3), BJ (16 build jobs: the cgroup has 27 CPUs / 31 GB), GATETO (5400), FULLTO (72000).
# Run: COMMIT=<sha> setsid nohup bash /root/pod/q3/podq3.sh > /root/pod/q3/podq3.out 2>&1 < /dev/null &
set -uo pipefail
COMMIT=${COMMIT:?COMMIT}; K=${K:-/root/ink}; Q3=${Q3:-/root/pod/q3}; BJ=${BJ:-16}; GATETO=${GATETO:-5400}; FULLTO=${FULLTO:-72000}
ST=$Q3/state; L=$Q3/podq3.log; mkdir -p $ST
say() { echo "$(date -u '+%F %T') podq3: $*" | tee -a $L; echo "$*" > $Q3/status; }
done_() { [ -e $ST/$1.done ]; }; mark() { touch $ST/$1.done; say "$1 done${2:+: $2}"; }
stopq() { [ -e $Q3/STOPQ ] && { say "STOPQ before $1"; exit 0; }; return 0; }
smoke_ok() { [ -e $K/state/ALLDONE ] && grep -q "ALLDONE: smoke" $K/podq.log; }
say "start (commit $COMMIT)"; export MCC_HOME=$HOME/.local/mcc
# --- smoke ---
stopq smoke
if ! done_ smoke; then
  p=$(pgrep -f '^bash /root/ink/kit/podq.sh$' | head -1)
  if [ -n "$p" ]; then say "smoke running (pid $p): waiting"; while kill -0 $p 2> /dev/null; do sleep 15; done
  elif ! smoke_ok; then say "smoke start"; COMMIT=$COMMIT K=$K timeout -k 60 7200 bash $K/kit/podq.sh > $K/podq.smoke.out 2>&1; fi
  if smoke_ok; then mark smoke "passed"; else say "smoke did not pass (no ALLDONE: smoke in $K/podq.log): FULL skipped"; touch $ST/smoke.failed; mark smoke "FAILED"; fi
fi
# --- gate ---
stopq gate
if ! done_ gate; then
  S=${COMMIT:0:12}; W=/root/gate/$S; mkdir -p /root/gate
  if [ ! -f $W/.built ]; then
    case "$W" in /root/gate/[0-9a-f]*) rm -rf -- "$W";; esac
    git clone -q /root/pod/ufsm $W && git -C $W fetch -q https://github.com/SuperOptimizer/ufsm master && git -C $W -c advice.detachedHead=false checkout -q $COMMIT \
      || { say "gate: clone / checkout $COMMIT failed"; mark gate "SETUP FAILED"; }
    if ! done_ gate; then say "gate: build ($BJ jobs)"
      ( cd $W && timeout -k 30 $GATETO flock -w $GATETO /root/gpu.lock bash -c "{ cmake --preset release && cmake --preset verify; } > build.log 2>&1 &&
          nice -n 5 cmake --build build -j $BJ --target ufsm test_sheet make_pipeline_fixture test_http_reader test_sampler_safety test_formats >> build.log 2>&1 &&
          nice -n 5 cmake --build build-verify -j $BJ --target suite-bins >> build.log 2>&1" ) && touch $W/.built || { say "gate: BUILD FAILED ($W/build.log)"; mark gate "BUILD FAILED"; }
    fi
  fi
  if ! done_ gate; then
    say "gate: warm the JIT cache, then the fast tier (all tests)"
    ( cd $W && bash scripts/toolchain/jit_warm.sh build-verify/test_unet 8 > jit.log 2>&1; rm -rf -- "$W/out"
      timeout -k 30 $GATETO flock -w $GATETO /root/gpu.lock timeout -k 30 $GATETO tests/run_suite.sh -t fast -o $W/out -B build-verify > $W/gate.out 2>&1 ); rc=$?
    ln -sfn $W /root/gate/last; mark gate "rc $rc ($W/gate.out; $(grep -E 'ALL PASSED|FAILED:|INCOMPLETE' $W/gate.out | tail -1 | cut -c1-200))"
  fi
fi
# --- full ---
stopq full
if ! done_ full; then
  if [ -e $ST/smoke.failed ]; then mark full "skipped (the smoke did not pass)"
  else
    [ -s $K/kit/podq.sh.new ] && bash -n $K/kit/podq.sh.new && mv -f $K/kit/podq.sh.new $K/kit/podq.sh && say "full: kit/podq.sh replaced ($(md5sum < $K/kit/podq.sh | cut -c1-8))"
    say "full start (COMMIT $COMMIT, FULL=1)"
    COMMIT=$COMMIT FULL=1 K=$K timeout -k 120 $FULLTO bash $K/kit/podq.sh > $K/podq.full.out 2>&1; rc=$?
    mark full "rc $rc ($(tail -1 $K/podq.log | cut -c1-200))"
  fi
fi
say "ALL DONE: the pod queue is empty"; touch $ST/ALLDONE

#!/usr/bin/env bash
# train's run-health warnings and start-up timeline (2026-11-20, devspeed + lead):
#  (1) the wind fixture has no ink labels: a --ink-head 1 run (--wd 0) leaves the ink head row without a gradient and prints
#      "ink bce 0.0000 (0% of windows)": by design, so neither "WARNING frozen:" nor "WARNING dead loss:" (a warning on every ink-only
#      run would train everyone to ignore it). Start-up lines carry "t+N.Ns " and one "startup:" summary follows the first step.
#  (2) --trunk-lr-mult 1e-9: the 3^3 convs get gradients but (lr x 1e-9) do not move: by step 100 "WARNING frozen:" names convs, not
#      every group (the head trains); --strict-health 1 exits 4 there.
#  (3) UFSM_STARTUP_TIMES=0: no timeline.
# Usage: test_health.sh UFSM MAKE_WIND_FIXTURE DIR
set -u
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf -- "$D"; mkdir -p "$D"; cd "$D" || exit 1
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1 || fail "fixture"
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 10 --ckpt-every 100000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --ink-head 1 --wd 0 --opt muon --muon-lr 0.002 --lr 0.001)
export UFSM_STARTUP_TIMES=1   # (run_suite exports 0: other tests grep train's lines anchored at ^)
"$U" train fx/sources.json --out a "${C[@]}" --steps 110 > a.log 2>&1 || fail "train ($D/a.log)"
grep -qE '^t\+[0-9]+\.[0-9]s ' a.log || fail "no t+ start-up prefix ($D/a.log)"
grep -qE '^(t\+[0-9.]+s )?startup: [0-9.]+ s to the first step.s end: .*sources [0-9.]+ s' a.log || fail "no startup summary ($D/a.log)"
grep -q 'ink bce 0.0000 (0% of windows' a.log || fail "the fixture run has no unlabelled ink term to stay quiet on ($D/a.log)"
! grep -q '^WARNING' a.log || fail "a warning on the unlabelled ink head: $(grep -m1 '^WARNING' a.log | cut -c1-120) ($D/a.log)"
"$U" train fx/sources.json --out b "${C[@]}" --steps 110 --trunk-lr-mult 1e-9 > b.log 2>&1 || fail "--trunk-lr-mult 1e-9 run ($D/b.log)"
grep -q '^WARNING frozen: .*conv L0' b.log || fail "no frozen warning for the trunk convs ($D/b.log)"
n=$(sed -nE 's/^WARNING frozen: ([0-9]+) of ([0-9]+) .*/\1 \2/p' b.log | head -1)
set -- $n; [ "${1:-0}" -ge 1 ] && [ "$1" -lt "${2:-0}" ] || fail "frozen count '$n' (want 1 <= n < groups) ($D/b.log)"
"$U" train fx/sources.json --out c "${C[@]}" --steps 110 --trunk-lr-mult 1e-9 --strict-health 1 > c.log 2>&1; rc=$?
[ "$rc" = 4 ] && grep -q -- '--strict-health 1: stopping' c.log || fail "--strict-health 1: rc $rc, want 4 ($D/c.log)"
UFSM_STARTUP_TIMES=0 "$U" train fx/sources.json --out e "${C[@]}" --steps 2 > e.log 2>&1 || fail "UFSM_STARTUP_TIMES=0 run ($D/e.log)"
! grep -qE '^t\+[0-9]|^startup:' e.log || fail "UFSM_STARTUP_TIMES=0 still prints the timeline ($D/e.log)"
echo "test_health: ok ($(grep -m1 '^WARNING frozen' b.log | cut -c1-100))"

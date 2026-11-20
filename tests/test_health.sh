#!/usr/bin/env bash
# train's run-health warnings and start-up timeline (2026-11-20, devspeed): on the wind fixture (no ink labels) a --ink-head 1 run with
# --wd 0 leaves the head's ink row without a gradient: by step 100 "WARNING frozen:" names a head row (not every group: the rest
# trains), and the ink term, exactly 0 for 100 steps, gives "WARNING dead loss:". --strict-health 1 exits 4 there. Start-up lines
# carry the "t+N.Ns " prefix and one "startup:" summary follows the first step; UFSM_STARTUP_TIMES=0 drops both.
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
grep -q '^WARNING frozen: .*head row' a.log || fail "no frozen head row warning ($D/a.log)"
n=$(sed -nE 's/^WARNING frozen: ([0-9]+) of ([0-9]+) .*/\1 \2/p' a.log | head -1)
set -- $n; [ "${1:-0}" -ge 1 ] && [ "$1" -lt "${2:-0}" ] || fail "frozen count '$n' (want 1 <= n < groups) ($D/a.log)"
grep -q '^WARNING dead loss: "ink bce"' a.log || fail "no dead-loss warning for the ink term ($D/a.log)"
"$U" train fx/sources.json --out b "${C[@]}" --steps 110 --strict-health 1 > b.log 2>&1; rc=$?
[ "$rc" = 4 ] && grep -q -- '--strict-health 1: stopping' b.log || fail "--strict-health 1: rc $rc, want 4 ($D/b.log)"
UFSM_STARTUP_TIMES=0 "$U" train fx/sources.json --out c "${C[@]}" --steps 2 > c.log 2>&1 || fail "UFSM_STARTUP_TIMES=0 run ($D/c.log)"
! grep -qE '^t\+[0-9]|^startup:' c.log || fail "UFSM_STARTUP_TIMES=0 still prints the timeline ($D/c.log)"
echo "test_health: ok ($(grep -m1 '^WARNING frozen' a.log | cut -c1-100))"

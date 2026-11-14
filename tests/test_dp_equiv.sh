#!/usr/bin/env bash
# test_dp_equiv (2026-11-14, ops audit #8): data parallel must train like gradient accumulation. Two ranks (train --gpus 0,1, or
# 0,0 = two logical devices on one card: src/ufsm_dev.cuh) at B 1 average two windows' gradients per update; one rank with --accum 2
# averages the same two windows (--det: the same batch sequence) before the same update. A process-wide global that should have been
# per device (the 11-13 / 11-14 bug class) makes the ranks disagree with the single-rank run: a different loss on the same windows, or
# updates that drift apart. Checked: each update's mean loss (DP step k vs the mean of accum windows 2k-1, 2k), equal to DPTOL at the
# first update (same weights, same windows: only the sum order differs) and within DPDRIFT after it (MX-fp4 stochastic rounding differs
# between a rank and a micro-step, so the trajectories separate slowly).
# Usage: test_dp_equiv.sh UFSM MAKE_WIND_FIXTURE DIR   (env UFSM_TEST_GPUS: the two-rank device list, default 0,1)
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3; G2=${UFSM_TEST_GPUS:-0,1}; N=${DPN:-4}
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(fx/sources-l1.json --task unwrap2 --P 64 --B 1 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2
   --warmup 0 --seed 5 --widths 16,32,64 --det 1 --opt muon --lr 1e-3 --muon-lr 0.01)
"$U" train "${C[@]}" --out dp --gpus "$G2" --steps "$N" > dp.log 2>&1 || { echo "FAIL: DP run ($D/dp.log)"; tail -5 dp.log; exit 1; }
"$U" train "${C[@]}" --out acc --gpus 0 --steps $((2 * N)) --accum 2 > acc.log 2>&1 || { echo "FAIL: --accum run ($D/acc.log)"; tail -5 acc.log; exit 1; }
grep -q 'x 2 GPU(s)' dp.log || { echo "FAIL: the DP run is not on two (logical) devices"; exit 1; }
python3 - dp.log acc.log "${DPTOL:-2e-3}" "${DPDRIFT:-0.05}" <<'PY'
import re, sys
def losses(p): return [float(m.group(2)) for m in (re.match(r'^step +(\d+) .*? loss ([0-9.eE+-]+)', l) for l in open(p)) if m]
dp, acc, tol, drift = losses(sys.argv[1]), losses(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
if len(dp) < 2 or len(acc) < 2 * len(dp): print('FAIL: step lines: DP %d, accum %d' % (len(dp), len(acc))); sys.exit(1)
bad = 0
for k, L in enumerate(dp):
    m = (acc[2 * k] + acc[2 * k + 1]) / 2; r = abs(L - m) / max(abs(m), 1e-9); lim = tol if k == 0 else drift
    print('update %d: DP loss %.5f, accum mean %.5f (windows %.5f %.5f): rel %.2e (limit %.0e)%s' % (k + 1, L, m, acc[2 * k], acc[2 * k + 1], r, lim, '' if r <= lim else '  FAIL'))
    bad += r > lim
print('test_dp_equiv: ' + ('ok' if not bad else '%d update(s) apart' % bad)); sys.exit(1 if bad else 0)
PY

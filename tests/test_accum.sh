#!/usr/bin/env bash
# train --accum K (gradient accumulation: K windows per optimizer update) on the wind fixture, --det:
#  (1) --accum 2 --steps 1 makes no update: its checkpoint's weights equal an --lr 0 run's, byte for byte;
#  (2) --accum 3 --steps 6 with UFSM_ACCUM_DUMP: each update's gradient is the mean of its 3 windows' gradients (max rel err < 1e-6;
#      exactness of the sum: = batch 3), every update at steps 3 and 6;
#  (3) --accum 4 --steps 8 logs "(update 2: 4 windows each)" and its loss is finite;
#  (4) --gns (gns.h): --accum 4 logs the gradient noise scale (|g|^2 at 1 vs 4 windows) on the line and in log.csv's gns columns; one GPU
#      without --accum and --gns 1 says there is no estimate and trains (the lr 0 run of (1)).
# Usage: test_accum.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --sched wsd --cooldown 0 --opt muon)
L=(--muon-lr 0.002 --lr 0.001)   # (opt: the first occurrence wins, so the lr pair goes before C)
fail() { echo "FAIL: $*"; exit 1; }
"$U" train fx/sources.json --out z --lr 0 --muon-lr 0 "${C[@]}" --steps 1 --gns 1 > z.log 2>&1 || fail "lr 0 ($D/z.log)"
"$U" train fx/sources.json --out a2 "${L[@]}" "${C[@]}" --steps 1 --accum 2 > a2.log 2>&1 || fail "accum 2, 1 step ($D/a2.log)"
mkdir -p dump; UFSM_ACCUM_DUMP=$PWD/dump "$U" train fx/sources.json --out a3 "${L[@]}" "${C[@]}" --steps 6 --accum 3 > a3.log 2>&1 || fail "accum 3 with dumps ($D/a3.log)"
"$U" train fx/sources.json --out a4 "${L[@]}" "${C[@]}" --steps 8 --accum 4 > a4.log 2>&1 || fail "accum 4 ($D/a4.log)"
grep -q '(update 2: 4 windows each)' a4.log || fail "no update line ($D/a4.log)"
grep '^step      8' a4.log | grep -qv nan || fail "accum 4: step 8 not finite"
grep -q -- '--gns: no estimate on one GPU' z.log || fail "--gns 1 on one GPU without --accum: no notice ($D/z.log)"
grep -q -- '--gns: gradient noise scale from |g|^2 at 1 vs 4 windows' a4.log || fail "accum 4: --gns not on by default ($D/a4.log)"
grep '^step      8' a4.log | grep -q ' gns ' || fail "accum 4: no gns on the step 8 line ($D/a4.log)"
head -1 a4/log.csv | grep -q ',gns,gns_g2,gns_s$' || fail "log.csv header without the gns columns"
awk -F, 'NR > 1 && $1 == 8 { exit !(NF == 15 && $14 != "") }' a4/log.csv || fail "log.csv step 8: no gns_g2 ($(tail -1 a4/log.csv))"
python3 -I - z/last.ckpt a2/last.ckpt <<'PY'
import sys, numpy as np
def w(p):   # the live parameters: the first of the 5 float32 blocks after the header line (p, ema, m, v, the Muon momentum)
    b = open(p, 'rb').read(); i = b.index(b'\n') + 1; n = (len(b) - i) // 20; return np.frombuffer(b[i:i + 4 * n], np.float32).astype(np.float64)
z, a2 = (w(p) for p in sys.argv[1:3]); n = min(len(z), len(a2)); z, a2 = z[:n], a2[:n]
d0 = np.count_nonzero(z != a2); print(f"  (1) accum 2 after 1 step vs lr 0: {d0} of {n} words differ")
import os
dd = 'dump'; worst = 0
for u in (3, 6):
    ws = [np.fromfile(f'{dd}/w{k}.f32', np.float32).astype(np.float64) for k in range(u - 2, u + 1)]; up = np.fromfile(f'{dd}/u{u}.f32', np.float32).astype(np.float64)
    ref = sum(ws) / 3; e = np.abs(up - ref).max() / max(np.abs(ref).max(), 1e-30); worst = max(worst, e)
    print(f"  (2) update {u // 3}: the mean of windows {u - 2}-{u}'s gradients vs the update's: max rel err {e:.2e} (|g| {np.linalg.norm(ref):.4g})")
r = worst
sys.exit(0 if d0 == 0 and r < 1e-6 else 1)
PY
echo "test_accum: ok"

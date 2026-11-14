#!/usr/bin/env bash
# The fp16 gradient scale of the head slabs (DESIGN 2026-11-14 "fp16 dead zone"), on the wind fixture, --det:
#  (1) one P 256 latent step at --gscale 1024, 65536 and 2^20 (lr 0, UFSM_HSG_ZERO=1, UFSM_LATG_DUMP): at 1024 a share of the supervised
#      voxels' fp16 logit gradients is 0 / subnormal, at 65536 fewer; the finished cell gradients / the scale against the 2^20 run's:
#      the 65536 run is closer than the 1024 run (the dead zone's error), and close (rel L2 < 1e-2);
#  (2) --gscale-grow 2 --gscale-max 4096: the scale doubles at steps 2 and 4 and stops; the checkpoint's "trainstate" holds 4096 and a
#      resume starts from it;
#  (3) the step lines carry the heads' KL group "(kl phase .. rv .. normal .. wind4 ..; phase error .. deg)": finite, >= 0 (to rounding),
#      below the bce they belong to.
# Usage: test_gscale.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
case "$D" in /*/*) ;; *) echo "test_gscale: DIR must be an absolute path below / ($D)"; exit 2;; esac
rm -rf -- "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 320 2 1 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
C=(--task unwrap2 --levels 1 --B 1 --gpus 0 --workers 2 --sampler-slots 2 --val-batches 0 --val-every 100000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --sched wsd --cooldown 0 --opt muon)
for gs in 1024 65536 1048576; do
  UFSM_HSG_ZERO=1 UFSM_LATG_DUMP=$PWD/latg_$gs.f16 "$U" train fx/sources-ink.json --out r$gs --lr 0 --muon-lr 0 "${C[@]}" --P 256 --steps 1 \
    --gscale $gs --gscale-grow 0 > r$gs.log 2>&1 || fail "gscale $gs ($D/r$gs.log)"
  grep -q "^hsg zero (GPU 0, gscale $gs" r$gs.log || fail "gscale $gs: no hsg zero line ($D/r$gs.log)"
  grep '^hsg zero' r$gs.log | sed 's/^/  /'
done
python3 -I - <<'PY'
import re, sys, numpy as np
def zr(gs):
    t = open(f'r{gs}.log').read(); m = re.search(r'phase (\d+) supervised: ([0-9.]+)% zero ([0-9.]+)% subnormal; rv (\d+): ([0-9.]+)% ([0-9.]+)%; wind4 (\d+): ([0-9.]+)% ([0-9.]+)%', t)
    return [float(x) for x in m.groups()]
a, b = zr(1024), zr(65536)
print(f"  (1) zero share at 1024 / 65536: phase {a[1]:.2f} / {b[1]:.2f} %, rv {a[4]:.2f} / {b[4]:.2f} %, wind4 {a[7]:.2f} / {b[7]:.2f} %")
ok = a[0] > 0 and a[6] > 0 and b[1] < a[1] and b[7] < a[7] and (a[1] > 0 or a[7] > 0)
g = {gs: np.fromfile(f'latg_{gs}.f16', np.float16).astype(np.float64) / gs for gs in (1024, 65536, 1048576)}
ref = g[1048576]; n = np.linalg.norm(ref)
e1, e2 = np.linalg.norm(g[1024] - ref) / n, np.linalg.norm(g[65536] - ref) / n
z1, z2 = np.mean((g[1024] == 0) & (ref != 0)), np.mean((g[65536] == 0) & (ref != 0))
print(f"  (1) cell gradients vs the 2^20 run: rel L2 {e1:.3e} at 1024, {e2:.3e} at 65536; cells zero where 2^20 is not: {100*z1:.2f} % / {100*z2:.2f} %")
ok = ok and e2 < e1 and e2 < 1e-2 and np.isfinite(ref).all()
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || fail "(1) dead zone numbers"
"$U" train fx/sources-ink.json --out gr --muon-lr 0.002 --lr 0.001 "${C[@]}" --P 64 --steps 6 --gscale 1024 --gscale-grow 2 --gscale-max 4096 > gr.log 2>&1 || fail "grow ($D/gr.log)"
grep -q '^step 2: grad scale -> 2048' gr.log && grep -q '^step 4: grad scale -> 4096' gr.log || fail "(2) no doubling at steps 2 / 4 ($D/gr.log)"
grep -q '^step 6: grad scale' gr.log && fail "(2) grew beyond --gscale-max"
head -1 gr/last.ckpt | grep -q '"trainstate":{"accum":1,"anchor":0,"upd":0,"gscale":4096}' || fail "(2) checkpoint trainstate ($(head -1 gr/last.ckpt | grep -o '"trainstate":{[^}]*}'))"
"$U" train fx/sources-ink.json --out gr --muon-lr 0.002 --lr 0.001 "${C[@]}" --P 64 --steps 8 --resume gr/last.ckpt --gscale-grow 0 > gr2.log 2>&1 || fail "grow resume ($D/gr2.log)"
grep -q 'gradient scale 4096 (the checkpoint' gr2.log || fail "(2) resume did not take the checkpoint's scale ($D/gr2.log)"
echo "  (2) grew 1024 -> 2048 -> 4096 at steps 2 / 4, checkpoint and resume keep 4096"
python3 -I - gr.log <<'PY'
import re, sys
bad = 0; n = 0
for ln in open(sys.argv[1]):
    if not ln.startswith('step '): continue
    m = re.search(r'\(kl ([^)]*)\)', ln)
    if not m: bad += 1; continue
    kl = dict(re.findall(r'(phase|rv|normal|wind4|ink) ([-0-9.]+)', m.group(1).split(';')[0]))
    bce = {k: float(v) for k, v in re.findall(r'(phase|rv|normal|wind4|ink) bce ([0-9.]+)', ln)}
    ang = re.search(r'phase error ([0-9.]+) deg', m.group(1))
    for k, v in kl.items():
        v = float(v); n += 1
        if not (v > -1e-3 and (k not in bce or v < bce[k])): bad += 1; print(f"  bad kl {k} {v} (bce {bce.get(k)}): {ln.strip()}")
    if not ang or not (0 <= float(ang.group(1)) <= 180): bad += 1
    last = m.group(0)
print(f"  (3) {n} KLs on the step lines, the last: {last}")
sys.exit(0 if n >= 6 * 4 and not bad else 1)
PY
[ $? -eq 0 ] || fail "(3) KL group ($D/gr.log)"
echo "test_gscale: ok"

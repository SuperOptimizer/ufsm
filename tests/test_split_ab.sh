#!/usr/bin/env bash
# The production loss path under --split z against one GPU (split audit 2026-11-14 #2): scripts/test/split_lat_ab.sh's trainer A/B in the
# suite, on the wind fixture (ink + fibre targets), with the live net's feature set in small: 6 levels, --split-top 3, 4 attention blocks with
# --attn-pe scroll, --gfilm 1, --film 2, --fstem 16, --gn-ema 1, --lat 4 --lat-guide 3, --rv 2, ink head + MIL, fibre head, --skip-empty 1,
# --fp4 2 with round to nearest and MX-fp8 gradients (UFSM_TEST_SR_OFF=1 UFSM_GRAD_MX4=0), --det 1. UFSM_GRAD_DUMP: the first step's loss
# and summed parameter gradient of: one GPU, the split (--gpus 0,1 with two cards, else both halves on GPU 0), one GPU again (the rerun
# noise). Pass: split-vs-one <= 2x rerun + a floor (loss 1e-7, gradients 1e-6; the DESIGN 11-13 A/B read 1.2e-7 at 1.0e-7 rerun) on the
# loss, the whole gradient and the gfilm A / B block alone (#1: the whole levels' A / B counted twice read far above that).
# Usage: test_split_ab.sh UFSM MAKE_WIND_FIXTURE DIR   (env P: window, default 256)
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 512 2 1 > fx.log 2>&1 || { echo "FAIL: fixture ($D/fx.log)"; exit 1; }
P=${P:-256}
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ -n "${CUDA_VISIBLE_DEVICES-}" ] && NG=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
SG=0,0; [ "$NG" -ge 2 ] && SG=0,1
C=(--task unwrap2 --P $P --B 1 --widths 16,16,32,32,48,48 --attn-blocks 4 --attn-pe scroll --gfilm 1 --film 2 --gn-ema 1 --skip-empty 1
   --lat 4 --lat-guide 3 --fstem 16 --fstem-drop 0 --rv 2 --rv-iso 100 --ink-head 1 --ink-mil 1 --fib-head 1
   --fp4 2 --opt muon --muon-lr 0.002 --lr 0.001 --warmup 0 --steps 3 --seed 3 --det 1 --workers 2 --sampler-slots 4 --val-batches 0
   --log-every 1 --ckpt-every 1000 --strict-sources 1)
fail() { echo "FAIL: $*"; exit 1; }
run() { local tag=$1; shift
  UFSM_GRAD_MX4=0 UFSM_TEST_SR_OFF=1 UFSM_GRAD_DUMP=$D/$tag.grad timeout -k 10 600 "$U" train fx/sources-ink.json --out $tag "${C[@]}" "$@" > $tag.log 2>&1 || true
  grep -q '^UFSM_GRAD_DUMP: step 1' $tag.log || fail "$tag: no gradient dump ($D/$tag.log)"; }
run one --gpus 0
run split --gpus $SG --split z --split-top 3
run one2 --gpus 0
GF=$(sed -nE 's/^UFSM_GRAD_DUMP: .*gfilm block from ([0-9]+).*/\1/p' one.log)
[ -n "$GF" ] || fail "no gfilm offset in the dump line ($D/one.log)"
python3 - "$D" "$GF" "$SG" <<'PY'
import sys, numpy as np
d, gf, sg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def ld(t):
    b = open(f"{d}/{t}.grad", "rb").read(); return np.frombuffer(b[:8], np.float64)[0], np.frombuffer(b[8:], np.float32).astype(np.float64)
l1, g1 = ld("one"); l2, g2 = ld("split"); l3, g3 = ld("one2")
if not (np.isfinite(g1).all() and np.isfinite(g2).all()): print("FAIL: non-finite gradients"); sys.exit(1)
r = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
ls, lr = abs(l2 - l1) / abs(l1), abs(l3 - l1) / abs(l1)
gs, grr = r(g2, g1), r(g3, g1)
fs, fr = r(g2[gf:], g1[gf:]), r(g3[gf:], g1[gf:])
ok = [ls <= 2 * lr + 1e-7, gs <= 2 * grr + 1e-6, gf < g1.size and np.linalg.norm(g1[gf:]) > 0 and fs <= 2 * fr + 1e-6]
print(f"test_split_ab (split on GPUs {sg}): loss {l1:.9g} / split {l2:.9g}: rel {ls:.2e} (rerun {lr:.2e}); gradient rel {gs:.2e} (rerun {grr:.2e}); "
      f"gfilm block ({g1.size - gf} params) rel {fs:.2e} (rerun {fr:.2e})")
for name, o in zip(["loss", "gradient", "gfilm block"], ok):
    if not o: print(f"FAIL: split vs one GPU, {name}")
sys.exit(0 if all(ok) else 1)
PY
echo "test_split_ab: ok"

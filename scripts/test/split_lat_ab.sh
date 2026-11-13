#!/usr/bin/env bash
# --lat 4 --split z A/B on one GPU (both halves on GPU 0): the first step's loss and summed gradient, split vs one GPU vs one GPU again.
# usage: splitab.sh OUTDIR [P]
U=${U:-$(dirname $0)/../../build/ufsm}
SRCJ=${SRCJ:-/vesuvius/ufsm/v2/lat/runs/unissl/sources-noink.json}
O=$1; P=${2:-256}; mkdir -p $O
HEADS=(--task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1)
EXTRA=(--lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-guide 3 --um-rungs 4.8,9.6:2,19.2
  --film 2 --fstem 16 --fstem-drop 0 --uw 1 --ink-head 1 --onfly-slots 1 --strict-sources 1 --det 1)
COMMON=($SRCJ --P $P --B 1 --widths ${WIDTHS:-32,32,64,64} ${ARCH} "${HEADS[@]}" --steps 5 --opt muon --fp4 2 --gpu-aug ${GAUG:-0} --gpu-input 1 --mem ${MEM:-auto}
  --val-batches 0 --workers 2 --sampler-slots 6 --warmup 0 --seed 3 --log-every 1 "${EXTRA[@]}")
run() { local tag=$1; shift; mkdir -p $O/$tag
  UFSM_GRAD_MX4=${GMX4:-0} UFSM_TEST_SR_OFF=${SROFF:-1} UFSM_GRAD_DUMP=$O/$tag.grad CUDA_VISIBLE_DEVICES=0 /home/forrest/bin/memcap $U train "${COMMON[@]}" --out $O/$tag "$@" > $O/$tag.log 2>&1
  echo "$tag rc $? $(grep UFSM_GRAD_DUMP $O/$tag.log)"; }
run one --gpus 0
run split --split z --gpus 0,0
run one2 --gpus 0
python3 - $O <<'EOF'
import sys, numpy as np
o = sys.argv[1]
def ld(t):
    b = open(f"{o}/{t}.grad", "rb").read(); return np.frombuffer(b[:8], np.float64)[0], np.frombuffer(b[8:], np.float32).astype(np.float64)
l1, g1 = ld("one"); l2, g2 = ld("split"); l3, g3 = ld("one2")
r = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
print(f"loss one {l1:.9g} split {l2:.9g} one2 {l3:.9g}: split rel {abs(l2-l1)/abs(l1):.2e}, rerun rel {abs(l3-l1)/abs(l1):.2e}")
print(f"grad: split vs one rel {r(g2, g1):.3e}, one2 vs one rel {r(g3, g1):.3e}, |g| {np.linalg.norm(g1):.4g}")
EOF

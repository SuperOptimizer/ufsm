#!/usr/bin/env bash
# split_finite.sh BIN CMDFILE [OUT]: the scratch2 NaN gate (perf2 2026-11-13). Step 1 of a trainer command line (CMDFILE: one line, "BIN train
# args...", e.g. the scratch2 launch's; P 1024 --split z --split-top 3 on both cards) with UFSM_GRAD_DUMP, under all six desk GPU locks; exit 0
# when every parameter gradient is finite, 1 with the non-finite count and its first / last index. a054edf (s2b weight streaming at 448
# channels, UFSM_S2B_WIDE=1) fails it: 5422144 NaN on enc3.c2 + level-3 film. The kernel alone at the same shape matches its reference
# (test_mx4), so no suite-sized repro exists; this gate needs the desk, real data and ~3 min. Extra env passes through (e.g. UFSM_S2B_WIDE=1).
set -u
U=${1:?BIN}; C=${2:?CMDFILE}; O=${3:-$(mktemp -d /vesuvius/ufsm/tmp-split-finite.XXXX)}; mkdir -p $O
read -ra A < $C; A=("${A[@]:2}")
for i in "${!A[@]}"; do case "${A[$i]}" in --out) A[$((i+1))]=$O/run;; --steps) A[$((i+1))]=2;; --gpu-aug) A[$((i+1))]=1;; --ckpt-every|--val-every) A[$((i+1))]=100000;; esac; done
B=(); skip=0; for a in "${A[@]}"; do   # gpu-aug 1 has no cube batches: --echo-max / --replay N would be refused ("needs --gpu-aug 2")
    if ((skip)); then skip=0; continue; fi; case "$a" in --echo-max|--replay) skip=1; continue;; esac; B+=("$a"); done; A=("${B[@]}")
LOCKS="/vesuvius/ufsm/opt/gpu0.lock /vesuvius/ufsm/opt/gpu1.lock /tmp/ufsm-suite-locks/gpu0.0 /tmp/ufsm-suite-locks/gpu0.1 /tmp/ufsm-suite-locks/gpu1.0 /tmp/ufsm-suite-locks/gpu1.1"
fd=9; for lk in $LOCKS; do eval "exec $fd> $lk"; flock -w 1800 $fd || { echo "lock $lk busy"; exit 2; }; fd=$((fd + 1)); done
env MALLOC_ARENA_MAX=4 UFSM_HEAD_SLAB_Z=8 CUDA_VISIBLE_DEVICES=0,1 UFSM_GRAD_DUMP=$O/grad.bin timeout -k 10 ${TMO:-300} ~/bin/memcap -t $U train "${A[@]}" > $O/train.log 2>&1
grep -q UFSM_GRAD_DUMP $O/train.log || { echo "no gradient dump (see $O/train.log)"; exit 2; }
python3 - $O/grad.bin <<'PY'
import sys, numpy as np
g = np.fromfile(sys.argv[1], dtype=np.float32)[2:]   # (an 8-byte header, then the fp32 gradients)
bad = np.flatnonzero(~np.isfinite(g))
if bad.size: print(f"split_finite: FAIL, {bad.size} non-finite gradients of {g.size} ([{bad[0]}, {bad[-1]}])"); sys.exit(1)
print(f"split_finite: ok, {g.size} gradients finite (max |g| {np.abs(g).max():.3g})")
PY

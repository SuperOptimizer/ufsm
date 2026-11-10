#!/bin/bash
# A latched CUDA error aborts training without a checkpoint: UFSM_TEST_CUDA_FAULT=2 makes a copy from a null host pointer fail after
# step 2's own check (where a validation pass's upload can fail); the save that follows must refuse (non-zero exit, the error and its
# call site printed) and last.ckpt must stay step 1's. Usage: test_cuda_abort.sh <ufsm binary> <make_wind_fixture binary>
set -u
U=$1 F=$2
D=$(mktemp -d /tmp/test_cuda_abort_XXXXXX)
fail() { echo "FAIL: $*"; echo "kept $D"; exit 1; }
"$F" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1 || fail "fixture"
C="--task unwrap2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --widths 16,32,64"
# shellcheck disable=SC2086
UFSM_TEST_CUDA_FAULT=2 "$U" train "$D/fx/sources.json" --out "$D/r" --steps 4 $C > "$D/log" 2>&1 && fail "train exited 0 after a CUDA error"
grep -q "unet_save: CUDA error .* at .*nn_core.cu:[0-9]* (cudaMemcpy" "$D/log" || fail "no 'unet_save: CUDA error ... at <site>' line ($D/log)"
[ -f "$D/r/last.ckpt" ] || fail "step 1's checkpoint missing"
head -c 400 "$D/r/last.ckpt" | grep -q '"step":1,' || fail "last.ckpt is not step 1's: $(head -c 400 "$D/r/last.ckpt" | grep -o '"step":[0-9]*')"
grep -q "^step      3" "$D/log" && fail "training went on past the error"
# the same run without the fault: exit 0, last.ckpt at step 4
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/ok" --steps 4 $C > "$D/ok.log" 2>&1 || fail "the run without the fault failed ($D/ok.log)"
head -c 400 "$D/ok/last.ckpt" | grep -q '"step":4,' || fail "the run without the fault did not save step 4"
rm -rf "$D"
echo "test_cuda_abort: ok"

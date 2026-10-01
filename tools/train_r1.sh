#!/bin/sh
# First real run: regenerate the sources file from the exported ground truth, then train on both GPUs at 128^3.
# (bf16 activations+gradients: 128^3 batch 2 takes 3.2 GB per GPU; 192^3 batch 1 takes 5.3 GB; voxel throughput is the same)
set -e
cd /home/forrest/ufsm
python3 tools/make_sources.py --out configs/all.json
RUN=${1:-runs/r1}
mkdir -p "$RUN"
exec ./build/ufsm train configs/all.json --out "$RUN" --gpus ${GPUS:-0,1} --P ${P:-128} --B ${B:-2} --prec ${PREC:-1} --steps ${STEPS:-20000} --lr 1e-3 --warmup 500 \
  --workers 12 --val-batches 16 --log-every 50 --val-every 500 --ckpt-every 1000

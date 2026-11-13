#!/usr/bin/env bash
# the on-the-fly pair-rule phase (src/band_onfly.h, sources "phase": {"onfly": {root, key, rmax}}) on the wind fixture's level-1 codes:
# phase-raster --pair 1 writes the stored phase; ufsm phase-onfly is byte-identical to it on the CPU and (with a GPU) on the GPU, also
# with small tiles (--budget-mb 64: many tiles per window, halo seams); and two --det training runs, one on the stored phase and one on
# the onfly phase, log the same losses. With a GPU every tile's codes are decoded on the device (volcomp lossless, k_ll_sub / k_ll_build)
# and the CPU-decoded path (UFSM_ONFLY_CPU_DECODE=1) is checked too
# Usage: test_phase_onfly.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
. "$(dirname "$(readlink -f "$0")")/det_cmp.sh"
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 > fx.log 2>&1
R=24   # (phase-raster --radius native voxels; --gs 2: rmax 12 codes voxels)
"$U" phase-raster fx/band 1 ph --axis fx/axis.json --gs 2 --pair 1 --radius $R --threads 4 --block 64 > pr.log 2>&1 || { echo "FAIL: phase-raster ($D/pr.log)"; exit 1; }
"$U" phase-onfly fx/band 1 --rmax 12 --ref ph 4.8 --windows 4 --P 80 --threads 4 > cpu.log 2>&1 || { echo "FAIL: CPU onfly differs ($D/cpu.log)"; tail -3 cpu.log; exit 1; }
"$U" phase-onfly fx/band 1 --rmax 12 --ref ph 4.8 --windows 4 --P 100 --budget-mb 64 --threads 4 > cpus.log 2>&1 || { echo "FAIL: CPU onfly, small tiles ($D/cpus.log)"; exit 1; }
grep -q 'byte-identical' cpu.log || { echo "FAIL: no byte-identical line"; exit 1; }
if nvidia-smi -L > /dev/null 2>&1; then
  "$U" phase-onfly fx/band 1 --rmax 12 --ref ph 4.8 --windows 4 --P 100 --gpu 0 --budget-mb 64 --threads 4 > gpu.log 2>&1 || { echo "FAIL: GPU onfly differs ($D/gpu.log)"; tail -3 gpu.log; exit 1; }
  grep -qE ' ([1-9][0-9]*) of \1 tiles decoded on the GPU' gpu.log || { echo "FAIL: not every tile's codes decoded on the GPU ($D/gpu.log)"; tail -1 gpu.log; exit 1; }
  UFSM_ONFLY_CPU_DECODE=1 "$U" phase-onfly fx/band 1 --rmax 12 --ref ph 4.8 --windows 4 --P 100 --gpu 0 --budget-mb 64 --threads 4 > gpuc.log 2>&1 || { echo "FAIL: GPU pair rule on CPU-decoded codes differs ($D/gpuc.log)"; exit 1; }
fi
# fused sets (sources "onfly": {"sets": [...], "tol"}): two phases of the codes (radius 24 and 8: on this fixture they agree, so every
# voxel takes the circular-mean path) fused by phase-fuse vs on the fly, CPU and GPU (the conflicts: PHerc0139 A + B in DESIGN.md)
"$U" phase-raster fx/band 1 ph8 --axis fx/axis.json --gs 2 --pair 1 --radius 8 --threads 4 --block 64 > pr8.log 2>&1 || { echo "FAIL: phase-raster r8 ($D/pr8.log)"; exit 1; }
"$U" phase-fuse fu 4.8 ph 4.8 ph8 4.8 --tol 0.05 --shard 128 > fu.log 2>&1 || { echo "FAIL: phase-fuse ($D/fu.log)"; exit 1; }
grep -qE 'fused [1-9]' fu.log || { echo "FAIL: no fused voxels ($D/fu.log)"; exit 1; }
for g in -1 0; do [ $g = 0 ] && ! nvidia-smi -L > /dev/null 2>&1 && continue
  "$U" phase-onfly fx/band 1 --rmax 12 --fuse fx/band 1 --fuse-rmax 4 --tol 0.05 --ref fu 4.8 --windows 4 --P 100 --gpu $g --budget-mb 64 --threads 4 > fu$g.log 2>&1 || { echo "FAIL: fused onfly differs (GPU $g, $D/fu$g.log)"; tail -2 fu$g.log; exit 1; }
done
# the codes pooled 2x on read ("pool2": 1, codes_pool2_open) == phase-raster --gs 4 (the pair rule on the 2x-pooled codes), CPU and GPU
"$U" phase-raster fx/band 1 pg4 --axis fx/axis.json --gs 4 --pair 1 --radius 48 --threads 4 --block 64 > pg4.log 2>&1 || { echo "FAIL: phase-raster --gs 4 ($D/pg4.log)"; exit 1; }
for g in -1 0; do [ $g = 0 ] && ! nvidia-smi -L > /dev/null 2>&1 && continue
  "$U" phase-onfly fx/band 1 --pool2 1 --rmax 12 --ref pg4 9.6 --windows 4 --P 48 --gpu $g --budget-mb 64 --threads 4 > p2$g.log 2>&1 || { echo "FAIL: pooled onfly differs from --gs 4 (GPU $g, $D/p2$g.log)"; tail -2 p2$g.log; exit 1; }
  # (GPU: the pooled view's tiles decoded and pooled on the device, codes_pool.h / k_codes_pool2; before 2026-11-12 on the CPU)
  [ $g = 0 ] && ! grep -qE ' ([1-9][0-9]*) of \1 tiles decoded on the GPU' p2$g.log && { echo "FAIL: pooled onfly tiles not all decoded on the GPU ($D/p2$g.log)"; exit 1; }
done
F=$(readlink -f fx); P=$(readlink -f ph)
python3 - "$F" "$P" <<'PY'
import json, sys
F, P = sys.argv[1:]
S = json.load(open(F + '/sources-l1.json'))
for s in S['sources']: s['targets']['band']['phase'] = {'root': P, 'key': '4.8', 'level': 1}; s['min_level'] = s['max_level'] = 1
json.dump(S, open('stored.json', 'w'))
for s in S['sources']: s['targets']['band']['phase'] = {'onfly': {'root': F + '/band', 'key': '1', 'rmax': 12}, 'level': 1}
json.dump(S, open('onfly.json', 'w'))
PY
C=(--task unwrap2 --levels 0,1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 4 --onfly-mb 64)
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "$U" train stored.json --out st "${C[@]}" > st.log 2>&1 || { echo "FAIL: stored-phase run ($D/st.log)"; exit 1; }
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "$U" train onfly.json --out of "${C[@]}" > of.log 2>&1 || { echo "FAIL: onfly run ($D/of.log)"; exit 1; }
grep -q 'on-the-fly phase on GPU 0' of.log || { echo "FAIL: no onfly config line ($D/of.log)"; exit 1; }
for f in st of; do grep '^step' $f.log | sed -E 's/ [0-9.]+ samp\/s.*//' > $f.txt; done
[ -s st.txt ] && det_cmp st.log of.log "stored vs onfly phase" || exit 1
echo "PASS: onfly phase byte-identical to phase-raster (CPU$( [ -f gpu.log ] && echo ', GPU')), training on it the same batches ($DET_CMP_N) and device inputs, losses within 2% ($(wc -l < st.txt) steps)"

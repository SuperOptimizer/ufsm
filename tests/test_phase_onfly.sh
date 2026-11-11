#!/usr/bin/env bash
# the on-the-fly pair-rule phase (src/band_onfly.h, sources "phase": {"onfly": {root, key, rmax}}) on the wind fixture's level-1 codes:
# phase-raster --pair 1 writes the stored phase; ufsm phase-onfly is byte-identical to it on the CPU and (with a GPU) on the GPU, also
# with small tiles (--budget-mb 64: many tiles per window, halo seams); and two --det training runs, one on the stored phase and one on
# the onfly phase, log the same losses
# Usage: test_phase_onfly.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
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
fi
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
"$U" train stored.json --out st "${C[@]}" > st.log 2>&1 || { echo "FAIL: stored-phase run ($D/st.log)"; exit 1; }
"$U" train onfly.json --out of "${C[@]}" > of.log 2>&1 || { echo "FAIL: onfly run ($D/of.log)"; exit 1; }
grep -q 'on-the-fly phase on GPU 0' of.log || { echo "FAIL: no onfly config line ($D/of.log)"; exit 1; }
for f in st of; do grep '^step' $f.log | sed -E 's/ [0-9.]+ samp\/s.*//' > $f.txt; done
[ -s st.txt ] && cmp -s st.txt of.txt || { echo "FAIL: stored vs onfly losses differ"; diff st.txt of.txt | head -4; exit 1; }
echo "PASS: onfly phase byte-identical to phase-raster (CPU$( [ -f gpu.log ] && echo ', GPU')), training on it identical ($(wc -l < st.txt) steps)"

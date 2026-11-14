#!/usr/bin/env bash
# the fibre planes on the fly (sources "fibre" targets, src/fiberraster.c fiber_onfly_open): fibre polylines (a UFIB of H / V fibres,
# straight and curved, crossing) over the wind fixture's CT; fiber-raster --ct writes the stored H / V planes (radius 2, guard 2, ring 2,
# iso 100); the planes' extent (draw cells) only near the fibres; ufsm fiber-onfly over random windows is byte-identical to them (GPU and, UFSM_FIBRE_CPU=1, CPU;
# the GPU within fibre_run's bound under the default fast math); and --fib-head training
# runs on the stored planes, the onfly planes (GPU) and the onfly planes (CPU) log the same losses (--det)
# Usage: test_fibre_onfly.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
. "$(dirname "$(readlink -f "$0")")/det_cmp.sh"
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
. "$(dirname "$(readlink -f "$0")")/fm_tol.sh"
B=${B:-$(dirname "$U")}   # (fm_strict reads $B/strict_math; the suite exports B, by hand: the ufsm binary's dir)
# fibre_run LOG WINDOWS P -- ARGS: fiber-onfly on the GPU vs the stored fiber-raster (host). Byte-identical under UFSM_STRICT_MATH; under the
# default fast math nn_fibre.cu cuts the 254 / 0 / 255 classes at radius / guard / ring thresholds on approximate sqrt, so a voxel on a
# threshold may flip: at most 1e-4 of the windows' voxels (both planes; summed "N bytes differ" of the window lines, fiber-onfly has
# no tolerance knob). The CPU path (UFSM_FIBRE_CPU=1) runs fiber-raster's own code and stays exact.
fibre_run() {
  local log=$1 w=$2 p=$3; shift 4
  if "$U" fiber-onfly "$@" > "$log" 2>&1; then grep -q 'byte-identical' "$log"; return; fi
  fm_strict && return 1
  grep -q 'DIFFERS from the stored raster' "$log" || return 1   # (any other failure)
  sed -nE 's/.*, ([0-9]+) bytes differ$/\1/p' "$log" | awk -v n="$((2 * p * p * p * w))" '{d += $1} END {printf "fast math: %d of <= %d voxels differ (allowed 1e-4)\n", d, n; exit !(d <= 1e-4 * n)}' >> "$log"
}
# det_cmp under strict math; else the device inputs and losses only (the batch hash covers the fibre planes: stored vs GPU on the fly)
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
python3 - <<'PY'
import math, struct
runs = []   # (family, [(z, y, x) level-0 voxels])
for k in range(6): runs.append(('H', [(40 + 30 * k, 60 + 25 * k + 8 * math.sin(t / 9), t) for t in range(-10, 270, 3)]))
for k in range(5): runs.append(('V', [(t, 50 + 35 * k, 70 + 30 * k + 6 * math.cos(t / 11)) for t in range(-10, 270, 2)]))
runs.append(('?', [(128, 128 + t, 128) for t in range(-60, 60, 4)]))
pts = [p for _, r in runs for p in r]; off = [0]
for _, r in runs: off.append(off[-1] + len(r))
with open('f.fib', 'wb') as f:
    f.write(b'UFIB' + struct.pack('<3i', 1, len(runs), len(pts)) + struct.pack('<%di' % len(off), *off) + bytes(ord(c) for c, _ in runs))
    for p in pts: f.write(struct.pack('<3f', *p))
PY
"$U" fiber-raster f.fib lab --um 9.6 --ct fx ct/9.6 --iso 100 --level 0 --radius 2 --guard 2 --ring 2 --shard 128 --threads 4 > fr.log 2>&1 || { echo "FAIL: fiber-raster ($D/fr.log)"; exit 1; }
O=(--ct fx ct/9.6 --level 0 --radius 2 --guard 2 --ring 2 --iso 100 --ref lab/H lab/V 9.6 --windows 6 --P 96)
fibre_run gpu.log 6 96 -- f.fib "${O[@]}" || { echo "FAIL: onfly (default) differs from fiber-raster ($D/gpu.log)"; tail -2 gpu.log; exit 1; }
UFSM_FIBRE_CPU=1 "$U" fiber-onfly f.fib "${O[@]}" > cpu.log 2>&1 || { echo "FAIL: onfly (CPU) differs ($D/cpu.log)"; tail -2 cpu.log; exit 1; }
UFSM_FIBRE_GPU_MVOX=0.05 fibre_run slab.log 6 96 -- f.fib "${O[@]}" || { echo "FAIL: onfly (GPU, 50k-voxel z slabs) differs ($D/slab.log)"; tail -2 slab.log; exit 1; }
grep -q 'byte-identical' cpu.log || { echo "FAIL: no byte-identical line"; exit 1; }
# the extent (z3_derived_extent: the draw cells): one short fibre in a corner marks only the chunks within its reach
python3 -c "
import struct; p = [(20, 20, 20 + t) for t in range(10)]
open('s.fib', 'wb').write(b'UFIB' + struct.pack('<3i', 1, 1, len(p)) + struct.pack('<2i', 0, len(p)) + b'H' + b''.join(struct.pack('<3f', *q) for q in p))"
"$U" fiber-onfly s.fib --ct fx ct/9.6 --level 0 --windows 1 --P 32 > ext.log 2>&1 || { echo "FAIL: fiber-onfly extent ($D/ext.log)"; exit 1; }
grep -qE 'the extent: 1 of ([2-9]|[1-9][0-9]+) chunks' ext.log || { echo "FAIL: the extent of one short fibre is not one chunk ($D/ext.log)"; grep extent ext.log; exit 1; }
python3 - "$PWD" <<'PY'
import json, sys
d = sys.argv[1]; S = json.load(open('fx/sources-fibhv.json'))
for k, pl in (('fibre_h', 'H'), ('fibre_v', 'V')): S['sources'][0]['targets'][k] = {'root': d + '/lab/' + pl, 'group': '.', 'encoding': 'prob254'}
json.dump(S, open('stored.json', 'w'))
for k, pl in (('fibre_h', 'H'), ('fibre_v', 'V')): S['sources'][0]['targets'][k] = {'fibre': {'afv': d + '/f.fib', 'plane': pl, 'radius': 2, 'guard': 2, 'ring': 2, 'iso': 100}}
json.dump(S, open('onfly.json', 'w'))
PY
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 4 --fib-head 1)
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "$U" train stored.json --out st "${C[@]}" > st.log 2>&1 || { echo "FAIL: stored run ($D/st.log)"; exit 1; }
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "$U" train onfly.json --out og "${C[@]}" > og.log 2>&1 || { echo "FAIL: onfly run ($D/og.log)"; exit 1; }
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 UFSM_FIBRE_CPU=1 "$U" train onfly.json --out oc "${C[@]}" > oc.log 2>&1 || { echo "FAIL: onfly CPU run ($D/oc.log)"; exit 1; }
# the planes' buffer sets shared by the reading threads (fiberraster.c fr_set, 2026-11-14): one set for four workers, the same batches
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 UFSM_FIBRE_SETS=1 "$U" train onfly.json --out o1 "${C[@]}" --workers 4 > o1.log 2>&1 || { echo "FAIL: onfly run, one buffer set ($D/o1.log)"; exit 1; }
for f in st og oc; do grep '^step' $f.log | sed -E 's/ [0-9.]+ samp\/s.*//' > $f.txt; done
grep -q 'fib bce 0\.[0-9]*' st.txt || { echo "FAIL: no fibre loss in the stored run"; exit 1; }
[ -s st.txt ] && fm_det_cmp st.log og.log "stored vs onfly fibre planes (GPU)" && det_cmp st.log oc.log "stored vs onfly fibre planes (CPU)" || exit 1
fm_det_cmp st.log o1.log "stored vs onfly fibre planes (one shared buffer set, 4 workers)" || exit 1
echo "PASS: the fibre planes on the fly byte-identical to fiber-raster (GPU, CPU), training on them the same batches and device inputs, losses within 2% ($(wc -l < st.txt) steps)"

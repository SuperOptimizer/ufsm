#!/usr/bin/env bash
# winding codes on the fly from q-assigned meshes (src/meshcodes.c mesh_codes_open, sources "onfly": {"meshes": ...}): four tifxyz
# meshes (two nested wavy sheets a turn apart, one sheet crossing both at another winding -> conflict voxels, one with invalid
# vertices and NaN q) rasterised by `ufsm raster --value q.tif --binary 0` at level 0 and level 1; `ufsm codes-onfly` over windows
# (not chunk aligned, several threads; the CPU stamp and the GPU one) must be byte-identical to both, and its extent must be a strict part of the chunk grid.
# Usage: test_codes_onfly.sh UFSM DIR
set -eu
U=$(readlink -f "$1"); D=$2
rm -rf "$D"; mkdir -p "$D/m"; cd "$D"
python3 - <<'PY'
import math, os, struct
def tif(path, w, h, vals):   # a minimal uncompressed float32 single-strip TIFF
    data = struct.pack('<%df' % (w * h), *vals); n = 9
    ifd = struct.pack('<H', n)
    for tag, typ, cnt, val in ((256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 32), (259, 3, 1, 1), (273, 4, 1, 8), (277, 3, 1, 1), (278, 4, 1, h), (279, 4, 1, len(data)), (339, 3, 1, 3)):
        ifd += struct.pack('<HHI', tag, typ, cnt) + (struct.pack('<HH', val, 0) if typ == 3 else struct.pack('<I', val))
    open(path, 'wb').write(b'II*\0' + struct.pack('<I', 8 + len(data)) + data + ifd + struct.pack('<I', 0))
def mesh(name, w, h, f):   # f(r, c) -> (x, y, z, q) or None
    d = 'm/%s.tifxyz' % name; os.makedirs(d); X, Y, Z, Q = [], [], [], []
    for r in range(h):
        for c in range(w):
            v = f(r, c)
            if v is None: X.append(-1.0); Y.append(-1.0); Z.append(-1.0); Q.append(float('nan'))
            else: X.append(v[0]); Y.append(v[1]); Z.append(v[2]); Q.append(v[3])
    for n, a in (('x', X), ('y', Y), ('z', Z), ('q', Q)): tif('%s/%s.tif' % (d, n), w, h, a)
    open(d + '/meta.json', 'w').write('{}')
S = 2.3   # vertex step in voxels (level 0): quads span several voxels, so the 0.5-voxel sampling matters
def sheet(rad, q0):
    def f(r, c):
        th = c * S / rad; rr = rad + 6 * math.sin(r / 17.0) + 3.0 * th
        return (150 + rr * math.cos(th), 150 + rr * math.sin(th), 12 + r * S, q0 + th / (2 * math.pi))
    return f
mesh('a100', 150, 100, sheet(60, 3.0)); mesh('b200', 180, 100, sheet(85, 4.0))
mesh('c300', 90, 100, lambda r, c: (40 + c * S, 150 + 0.4 * c + 5 * math.sin(r / 9.0), 12 + r * S, 9.5 + c / 400.0))   # a plane through both sheets
def holes(r, c):
    if (r // 13 + c // 11) % 4 == 0: return None
    v = sheet(110, 5.0)(r, c); return v if (r + c) % 29 else (v[0], v[1], v[2], float('nan'))
mesh('d400', 200, 100, holes)
PY
M=$(ls -d m/*.tifxyz)
for L in 0 1; do
  "$U" raster st$L --shape 256,320,320 --um 2.4 --level $L --binary 0 --levels 1 --threads 4 --shard 128 --value q.tif $M > raster$L.log 2>&1 || { echo "FAIL: ufsm raster level $L ($D/raster$L.log)"; exit 1; }
  K=$(ls st$L | grep -v zarr.json)
  for T in 1 5; do
    "$U" codes-onfly m --ct st$L $K --level $L --ref st$L $K --windows 10 --P $((96 >> L)) --threads $T --seed $((L + T)) > on$L.$T.log 2>&1 || { echo "FAIL: codes-onfly differs from ufsm raster at level $L, $T threads ($D/on$L.$T.log)"; tail -2 on$L.$T.log; exit 1; }
    grep -q 'byte-identical' on$L.$T.log || { echo "FAIL: no byte-identical line ($D/on$L.$T.log)"; exit 1; }
  done
  # the whole volume in one read
  "$U" codes-onfly m --ct st$L $K --level $L --ref st$L $K --windows 1 --P $((256 >> L)) --threads 4 > all$L.log 2>&1 || { echo "FAIL: the whole-volume read differs at level $L ($D/all$L.log)"; exit 1; }
  # the device stamp (src/nn_mcodes.cu): the same windows and the whole volume byte-identical to the raster, every read on the GPU; a
  # 1 MB key slab forces many z slabs per read (the slab seams)
  for G in "" "UFSM_MESHCODES_GPU_MB=1"; do
    env $G "$U" codes-onfly m --ct st$L $K --level $L --ref st$L $K --windows 10 --P $((96 >> L)) --threads 5 --seed $((L + 5)) --gpu 0 > gpu$L.log 2>&1 || { echo "FAIL: the GPU stamp differs from ufsm raster at level $L $G ($D/gpu$L.log)"; exit 1; }
    env $G "$U" codes-onfly m --ct st$L $K --level $L --ref st$L $K --windows 1 --P $((256 >> L)) --threads 4 --gpu 0 > gpuall$L.log 2>&1 || { echo "FAIL: the GPU whole-volume read differs at level $L $G ($D/gpuall$L.log)"; exit 1; }
    for f in gpu$L.log gpuall$L.log; do grep -q 'byte-identical' $f && grep -q 'on the CPU 0 ' $f && grep -q 'device failures 0' $f || { echo "FAIL: not every read ran on the GPU, or not identical ($D/$f)"; exit 1; }; done
  done
done
# the first open above loaded the TIFFs and wrote the pack (m/.meshcodes-*.pack), the later ones mmap'd it; without a pack, and with a
# stale one (a mesh's q.tif touched -> another key), the result is the same
ls m/.meshcodes-*.pack > /dev/null 2>&1 || { echo "FAIL: no mesh pack was written"; exit 1; }
UFSM_MESHCODES_CACHE=off "$U" codes-onfly m --ct st0 $(ls st0 | grep -v zarr.json) --level 0 --ref st0 $(ls st0 | grep -v zarr.json) --windows 4 --P 96 > nopack.log 2>&1 || { echo "FAIL: without the pack ($D/nopack.log)"; exit 1; }
touch -d '2001-01-01' m/a100.tifxyz/q.tif
"$U" codes-onfly m --ct st0 $(ls st0 | grep -v zarr.json) --level 0 --ref st0 $(ls st0 | grep -v zarr.json) --windows 4 --P 96 > stale.log 2>&1 || { echo "FAIL: after a changed input ($D/stale.log)"; exit 1; }
[ "$(ls m/.meshcodes-*.pack | wc -l)" = 2 ] || { echo "FAIL: a changed input did not make a new pack"; exit 1; }
grep -qE '\([1-9][0-9]* conflict\)' all0.log || { echo "FAIL: the fixture made no conflict voxels ($D/all0.log)"; exit 1; }
read -r a b <<< "$(grep -o 'the extent: [0-9]* of [0-9]*' on0.1.log | awk '{print $3, $5}')"
[ "$a" -gt 0 ] && [ "$a" -lt "$b" ] || { echo "FAIL: the extent $a of $b chunks is not a strict part"; exit 1; }
echo "test_codes_onfly: OK (levels 0 / 1, 1 / 5 threads, whole volume, CPU and GPU stamps; extent $a of $b chunks; $(grep -o '[0-9]* conflict' all0.log | head -1))"

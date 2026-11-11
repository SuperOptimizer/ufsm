#!/usr/bin/env python3
"""shardsel.py ARRAY_DIR SCALE BOXES.tsv SCROLL[,SCROLL..] [MARGIN]: the shard files of a sharded zarr v3 array (ARRAY_DIR/zarr.json, chunks
under c/z/y/x) that intersect the named scrolls' wbench boxes (boxes.tsv rows in their level's voxels) scaled by SCALE (array voxels per box
voxel, e.g. 2 for 4.8 um codes under 9.6 um boxes, 0.5 for 19.2 um labels), +- MARGIN box voxels (default 64). Prints paths relative to
ARRAY_DIR (zarr.json first) and the total bytes on stderr: the eval kit copies only these (univ eval harness, 2026-11-11)."""
import json, os, sys
a, sc, bt, scs = sys.argv[1], float(sys.argv[2]), sys.argv[3], set(sys.argv[4].split(',')); mg = int(sys.argv[5]) if len(sys.argv) > 5 else 64
j = json.load(open(os.path.join(a, 'zarr.json'))); sh = j['chunk_grid']['configuration']['chunk_shape']; shape = j['shape']
sel = set()
for l in open(bt):
    if l.startswith('#') or not l.strip(): continue
    f = l.rstrip('\n').split('\t')
    if f[1] not in scs: continue
    o = [int(v) for v in f[2:5]]; n = [int(v) for v in f[5:8]]
    lo = [max(0, int((o[d] - mg) * sc) // sh[d]) for d in range(3)]
    hi = [min((shape[d] - 1) // sh[d], int((o[d] + n[d] + mg) * sc) // sh[d]) for d in range(3)]
    for z in range(lo[0], hi[0] + 1):
        for y in range(lo[1], hi[1] + 1):
            for x in range(lo[2], hi[2] + 1): sel.add((z, y, x))
tot = os.path.getsize(os.path.join(a, 'zarr.json')); out = ['zarr.json']
for z, y, x in sorted(sel):
    p = 'c/%d/%d/%d' % (z, y, x)
    if os.path.exists(os.path.join(a, p)): out.append(p); tot += os.path.getsize(os.path.join(a, p))
print('\n'.join(out)); print('%s: %d shards, %.2f GB' % (a, len(out) - 1, tot / 1e9), file=sys.stderr)

#!/usr/bin/env python3
"""ctcover.py LOCALROOT LOCALKEY MIRROR MIRRORKEY LEVEL z,y,x,nz,ny,nx [MARGIN] [CACHE]: is a box fully readable from the local CT copy?

The local desk copies (/vesuvius/ufsm/v2/ct/<scroll>/<scan>.zarr) may hold only some shards of a level. A shard missing locally is fine when the
mirror does not have it either (an all-air shard, absent in both), and not fine when the mirror has it. Prints "local" (exit 0) or "mirror"
(exit 1, with the first missing shard on stderr). The box is in LEVEL voxels; MARGIN (default 640: one window) widens it to every shard a
sliding-window predict over the box can read. Mirror answers (HEAD per shard) are cached in CACHE (default ~/.cache/ufsm-ctcover.tsv)."""
import json, os, sys, urllib.request

lroot, lkey, mroot, mkey, lv, box = sys.argv[1:7]
margin = int(sys.argv[7]) if len(sys.argv) > 7 else 640
cache = sys.argv[8] if len(sys.argv) > 8 else os.path.expanduser('~/.cache/ufsm-ctcover.tsv')
b = [int(v) for v in box.split(',')]
ldir = os.path.join(lroot, lkey, lv)
if not os.path.exists(os.path.join(ldir, 'zarr.json')):
    print('mirror'); print(f'ctcover: no local level {ldir}', file=sys.stderr); sys.exit(1)
m = json.load(open(os.path.join(ldir, 'zarr.json')))
shape, sh = m['shape'], m['chunk_grid']['configuration']['chunk_shape']
seen = {}
if os.path.exists(cache):
    for line in open(cache):
        k, v = line.rstrip('\n').split('\t'); seen[k] = v
rng = []
for d in range(3):
    lo, hi = max(0, b[d] - margin), min(shape[d], b[d] + b[d + 3] + margin)
    rng.append(range(lo // sh[d], (hi + sh[d] - 1) // sh[d]))
new = []
for i in rng[0]:
    for j in rng[1]:
        for k in rng[2]:
            if os.path.exists(os.path.join(ldir, 'c', str(i), str(j), str(k))): continue
            url = f'{mroot}/{mkey}/{lv}/c/{i}/{j}/{k}'
            if url not in seen:
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r: seen[url] = 'yes' if r.status == 200 else 'no'
                except urllib.error.HTTPError as e: seen[url] = 'no' if e.code == 404 else 'err'
                except Exception: seen[url] = 'err'
                new.append(url)
            if seen[url] != 'no':
                with open(cache, 'a') as f:
                    for u in new: f.write(f'{u}\t{seen[u]}\n')
                print('mirror'); print(f'ctcover: {lkey} level {lv} shard {i}/{j}/{k} missing locally, mirror: {seen[url]}', file=sys.stderr); sys.exit(1)
with open(cache, 'a') as f:
    for u in new: f.write(f'{u}\t{seen[u]}\n')
print('local')

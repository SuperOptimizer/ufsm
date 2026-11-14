#!/usr/bin/env python3
"""scripts/holdouts/wbcover.py [BOXES.TSV] [HOLDOUTS.JSON]: every wbench box (scripts/wbench/boxes.tsv, CT voxels of its level) must lie inside
a held-out box (holdouts-20261114.json, level-0 voxels of a named CT) of its scroll, or no training window keeps off it. Prints the boxes
that are not covered and exits 1 if any are (audit eval#10: scripts/wbench/holdouts.json held only C1-C24, so 46 of the 70 1667 boxes,
8 of the 12 core guard boxes among them, were in no holdout box for a source relying on holdout_file alone).
A box is covered when one held-out box on the scroll's training scan (the CT names wbench.sh BOXES=holdout matches) contains it whole."""
import json, os, sys

H = os.path.dirname(os.path.abspath(__file__))
bt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, '..', 'wbench', 'boxes.tsv')
hj = sys.argv[2] if len(sys.argv) > 2 else os.path.join(H, 'holdouts-20261114.json')
HB = [b for b in json.load(open(hj))['boxes'] if b.get('score', 1) != 0]
scan = {'paris4': '20260411134726-2.400um', '1667': '20251217075048-2.399um', 'PHerc0139': '20250728140407-9.362um', 'PHerc0500P2': '20250820143440-9.362um',
        'PHerc0814': '20250804134230-9.362um'}   # (wbench.sh BOXES=holdout's map; paris4j is the 2023 scan, no training scan: not checked)
bad = 0; n = 0; skip = 0
for l in open(bt):
    if l.startswith('#') or not l.strip(): continue
    f = l.rstrip('\n').split('\t'); name, sc = f[0], f[1]; z, y, x, nz, ny, nx, lv = (int(v) for v in f[2:9]); s = 1 << lv
    if sc not in scan: skip += 1; continue
    lo = (z * s, y * s, x * s); hi = ((z + nz) * s, (y + ny) * s, (x + nx) * s); n += 1
    cand = [b for b in HB if scan[sc] in b.get('ct', '')]
    ok = any(all(b['l0'][d] <= lo[d] and hi[d] <= b['l0'][d] + b['l0'][3 + d] for d in range(3)) for b in cand)
    if not ok:
        bad += 1; print(f'not held out: {name} ({sc}) level-0 [{lo[0]}, {hi[0]}) x [{lo[1]}, {hi[1]}) x [{lo[2]}, {hi[2]}) ({len(cand)} held-out boxes on that scan)')
if skip: print(f'({skip} boxes on scans that are not training scans, e.g. paris4j: not checked)')
print(f'{n - bad} of {n} wbench boxes inside a held-out box of {os.path.basename(hj)}')
sys.exit(1 if bad else 0)

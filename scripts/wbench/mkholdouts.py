#!/usr/bin/env python3
"""scripts/wbench/mkholdouts.py [--check] [--boxes boxes.tsv] [--holdouts holdouts.json] [--out FILE]

Coverage of the wbench eval boxes by the training holdouts (eval audit 2026-11-14 #10). Every boxes.tsv row (level-L voxels of its scroll's
CT) must lie inside one holdouts.json box of the same CT (level-0 voxels); holdouts.json had stopped at C24 while boxes.tsv grew to C70, so
46 of the 70 1667 boxes (8 of the 12 core guard boxes) were held out of nothing for any source relying on "holdout_file" alone.

--check: print every uncovered box and exit 1 if there is one (0 otherwise). Without --check: write holdouts.json (or --out) with one box
per uncovered row appended (name, scroll, ct, l0 = the row's box times 2^level); existing boxes are kept as they are (some are wider
than their eval box on purpose, e.g. 139-1 / 814-1, the dino holdouts). The CT of a scroll is taken from the boxes it already has.
"""
import json
import os
import sys


def load_rows(path):
    rows = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        lv = int(f[8])
        rows.append({'name': f[0], 'scroll': f[1], 'l0': [int(v) << lv for v in f[2:8]], 'note': f[9] if len(f) > 9 else ''})
    return rows


def inside(a, b):   # box a (z,y,x,nz,ny,nx) inside box b
    return all(b[d] <= a[d] and a[d] + a[3 + d] <= b[d] + b[3 + d] for d in range(3))


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    opt = {'--boxes': os.path.join(here, 'boxes.tsv'), '--holdouts': os.path.join(here, 'holdouts.json'), '--out': None}
    check = '--check' in argv
    args = [a for a in argv if a != '--check']
    for i in range(0, len(args) - 1, 2):
        if args[i] not in opt:
            sys.exit(f'mkholdouts: unknown option {args[i]}')
        opt[args[i]] = args[i + 1]
    H = json.load(open(opt['--holdouts']))
    ct_of = {}
    for b in H['boxes']:
        ct_of.setdefault(b['scroll'], b['ct'])
    ct_of.setdefault('paris4j', ct_of.get('paris4'))
    missing = []
    for r in load_rows(opt['--boxes']):
        ct = ct_of.get(r['scroll'])
        if not ct:
            sys.exit(f"mkholdouts: no CT known for scroll {r['scroll']} (box {r['name']}): add one of its boxes to the holdouts by hand")
        if not any(os.path.basename(b['ct'].rstrip('/')) == os.path.basename(ct.rstrip('/')) and inside(r['l0'], b['l0']) for b in H['boxes']):
            missing.append((r, ct))
    for r, ct in missing:
        print(f"uncovered: {r['name']} {r['scroll']} l0 {r['l0']}", file=sys.stderr)
    if check:
        print(f"mkholdouts: {len(missing)} of {len(load_rows(opt['--boxes']))} eval boxes not inside a holdout box", file=sys.stderr)
        return 1 if missing else 0
    for r, ct in missing:
        H['boxes'].append({'name': r['name'], 'scroll': 'paris4' if r['scroll'] == 'paris4j' else r['scroll'], 'ct': ct, 'l0': r['l0']})
    if missing:
        H['version'] = H.get('version', 'wbench') + '+mkholdouts-20261114'
    out = opt['--out'] or opt['--holdouts']
    with open(out + '.tmp', 'w') as f:
        json.dump(H, f, indent=2)
        f.write('\n')
    os.replace(out + '.tmp', out)
    print(f"mkholdouts: {len(missing)} boxes added -> {out} ({len(H['boxes'])} boxes)", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

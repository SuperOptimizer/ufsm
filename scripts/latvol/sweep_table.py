#!/usr/bin/env python3
"""sweep_table.py RES_DIR RATES.tsv [OUT.tsv]: the compression sweep's inference table. Per checkpoint tag and arm: the rate (bytes per
source voxel of the re-encoded box regions, all scans), the mean p05x / ids over the boxes, and the paired difference to the top anchor
(arm orig: the upstream uncompressed CT) with a 95% bootstrap interval over boxes. Rows: RES_DIR/TAG-ARM.tsv (wbench)."""
import sys, os, glob, csv, random
res, ratef = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else None
rate = {}
for l in open(ratef):
    a, s, b, v = l.split('\t'); r = rate.setdefault(a, [0.0, 0.0]); r[0] += float(b); r[1] += float(v)
rate = {a: b / v for a, (b, v) in rate.items()}; rate['raw'] = 1.0
def rows(f):
    d = {}
    for r in csv.DictReader(open(f), delimiter='\t'):
        try: d[r['box']] = (float(r['p05x']), float(r['ids']))
        except ValueError: pass
    return d
def ci(d, rng):
    n = len(d); m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000)); return m[100], m[3899]
tags = sorted({os.path.basename(f).rsplit('-', 1)[0] for f in glob.glob(res + '/*-orig.tsv')})
lines = ['tag\tarm\tcodec\tB_per_vox\tn\tp05x\td_p05x\tlo\thi\tids\td_ids\tlo\thi']
for t in tags:
    A = rows(f'{res}/{t}-orig.tsv')
    arms = sorted((os.path.basename(f)[len(t) + 1:-4] for f in glob.glob(f'{res}/{t}-*.tsv')), key=lambda a: -rate.get(a, 0))
    for a in arms:
        if a not in rate: continue
        B = rows(f'{res}/{t}-{a}.tsv'); bx = [b for b in A if b in B]
        if not bx: continue
        rng = random.Random(1); o = []
        for k in (0, 1):
            d = [B[b][k] - A[b][k] for b in bx]; lo, hi = ci(d, rng); o += [sum(B[b][k] for b in bx) / len(bx), sum(d) / len(d), lo, hi]
        lines.append(f"{t}\t{a}\t{'anchor' if a == 'orig' else 'volcomp' if a[0] == 'v' else 'latvol'}\t{rate[a]:.5f}\t{len(bx)}\t" + '\t'.join(f'{x:+.4f}' if i % 4 else f'{x:.4f}' for i, x in enumerate(o)))
print('\n'.join(lines).expandtabs(11))
if out: open(out, 'w').write('\n'.join(lines) + '\n')

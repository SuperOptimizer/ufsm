#!/usr/bin/env python3
"""xres_pair.py XRES.tsv NATIVE.tsv: cross-res pairs (box xNAME on the fine scan vs NAME on the 9.36 um scan, same checkpoint):
per-metric mean of fine - native with a paired bootstrap 95 % CI (10000 resamples, seed 1), all boxes and the boxes with ceil >= 0.8."""
import sys, random
def rows(p):
    L = [l.rstrip('\n').split('\t') for l in open(p) if l.strip()]; h = L[0]; return {r[1]: dict(zip(h, r)) for r in L[1:]}
A, B = rows(sys.argv[1]), rows(sys.argv[2]); random.seed(1)
for met in ('p05x', 'ids', 'p1x'):
    for clean in (0, 1):
        pr = []
        for k, a in A.items():
            b = B.get(k[1:]) if k.startswith('x') else None
            if not b: continue
            try: x, y = float(a[met]), float(b[met]); c = float(b['ceil'])
            except (ValueError, KeyError): continue
            if clean and c < 0.8: continue
            pr.append((x, y))
        if not pr: continue
        d = [x - y for x, y in pr]; m = sum(d) / len(d)
        bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(10000))
        print('%-5s %s n %2d | fine %.4f native %.4f | fine-native %+.4f [%+.4f, %+.4f]' % (met, 'ceil>=0.8' if clean else 'all      ', len(d),
              sum(x for x, _ in pr) / len(pr), sum(y for _, y in pr) / len(pr), m, bs[250], bs[9749]))

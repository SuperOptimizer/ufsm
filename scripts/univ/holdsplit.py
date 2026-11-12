#!/usr/bin/env python3
"""holdsplit.py REJECT.txt A.tsv:B.tsv [A2.tsv:B2.tsv ...]: the masked-holdout control (2026-11-11). U1 trains on windows over the wbench
boxes with their labels masked (box + 16-voxel guard: the CT is seen), except the boxes named in REJECT.txt, whose windows are rejected
(CT unseen). If seeing the CT helped, A - B (the universal net minus a baseline that never drew any of these boxes) would be larger on
the masked boxes than on the reject ones. Rows of the wbench TSV pairs are pooled (e.g. the core rows and the native 9.36 um rows);
per metric: A - B on each group with a paired bootstrap 95 % CI, and masked - reject of those means (groups resampled independently;
10000 resamples, seed 1). All boxes and the boxes with ceil >= 0.8 (few reject boxes are clean: read the n).
The groups differ in scrolls (reject: HOLD + 0139; masked: mostly Paris 4 / 1667), so masked - reject is not zero without any masking:
u1-3000-prefix (trained with every holdout window rejected) - lat-wz896p gives +0.018 p05x / +0.049 ids. Compare against that."""
import sys, random
def rows(p):
    L = [l.rstrip('\n').split('\t') for l in open(p) if l.strip()]; h = L[0]; return {r[1]: dict(zip(h, r)) for r in L[1:]}
rej = {l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')}
P = []
for ab in sys.argv[2:]:
    a, b = ab.split(':'); A, B = rows(a), rows(b); P += [(k, A[k], B[k]) for k in A if k in B]
random.seed(1)
def boot(d): return sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(10000))
for met in ('p05x', 'ids'):
    for clean in (0, 1):
        g = {0: [], 1: []}
        for k, a, b in P:
            try: x, y, c = float(a[met]), float(b[met]), float(b['ceil'])
            except (ValueError, KeyError): continue
            if clean and c < 0.8: continue
            g[k in rej].append(x - y)
        tag = 'ceil>=0.8' if clean else 'all      '; bs = {}
        for r in (0, 1):
            d = g[r]
            if not d: print('%-5s %s %s n  0' % (met, tag, 'reject' if r else 'masked')); continue
            bs[r] = boot(d); print('%-5s %s %s n %2d | A-B %+.4f [%+.4f, %+.4f]' % (met, tag, 'reject' if r else 'masked', len(d), sum(d) / len(d), bs[r][250], bs[r][9749]))
        if len(bs) == 2:
            random.shuffle(bs[0]); random.shuffle(bs[1]); dd = sorted(x - y for x, y in zip(bs[0], bs[1]))
            print('%-5s %s masked - reject %+.4f [%+.4f, %+.4f]%s' % (met, tag, sum(g[0]) / len(g[0]) - sum(g[1]) / len(g[1]), dd[250], dd[9749],
                  '  (n reject %d: wide)' % len(g[1]) if len(g[1]) < 5 else ''))

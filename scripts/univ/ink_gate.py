#!/usr/bin/env python3
"""ink_gate.py VSCORES.tsv [BASE.tsv] [--min-gain G]: per-scroll 3-D crop AUC (ink-vscore rows of eval.sh: crop P N known_frac ink_share auc ...)
and a gate (audit inkmil #1). Per-scroll means are reported separately; the headline is the mean over the scrolls that can move (Paris 4, 0500P2,
0841, 0139); PHerc1667 is the floor scroll for every model we have (ink3d-ext teacher 0.527 at 2.4 um) and is only a non-regression check
(>= base - 0.03). With BASE: no scroll may fall by more than 0.03 (1667: noise), headline must not fall. Exit 1 on a failed gate.
Not a gate: the banner AUC against the published 1667 ink (a model's output, wrong-way strip: urender#1 / #3)."""
import sys, re
def load(p):
    d = {}
    for l in open(p):
        if l.startswith('#') or not l.strip(): continue
        f = l.split('\t') if '\t' in l else l.split()
        if len(f) < 6: continue
        try: auc = float(f[5])
        except ValueError: continue
        c = f[0]; s = re.match(r'(PHerc\w+?)_(w\w+|wa\d)', c); s = s.group(1) if s else 'PHercParis4'
        d.setdefault(s, []).append(auc)
    return {k: sum(v) / len(v) for k, v in d.items()}, {k: len(v) for k, v in d.items()}
a = [x for x in sys.argv[1:] if not x.startswith('--')]
if not a: sys.exit(__doc__)
cur, n = load(a[0]); base = load(a[1])[0] if len(a) > 1 else {}
HEAD = ('PHercParis4', 'PHerc0500P2', 'PHerc0841', 'PHerc0139'); fail = 0
print('%-12s %3s %7s %7s %7s' % ('scroll', 'n', 'auc', 'base', 'delta'))
for k in sorted(cur):
    b = base.get(k); dl = cur[k] - b if b is not None else None
    bad = dl is not None and dl < -0.03; fail |= bad
    print('%-12s %3d %7.3f %7s %7s %s%s' % (k, n[k], cur[k], '%.3f' % b if b is not None else '-', '%+.3f' % dl if dl is not None else '-', '  (floor scroll: non-regression only)' if k == 'PHerc1667' else '', '  FAIL' if bad else ''))
h = [cur[k] for k in HEAD if k in cur]; hb = [base[k] for k in HEAD if k in base]
if h: print('headline (mean of %s): %.3f%s' % (','.join(k.replace('PHerc', '') for k in HEAD if k in cur), sum(h) / len(h), ' base %.3f' % (sum(hb) / len(hb)) if hb else ''))
if h and hb and sum(h) / len(h) < sum(hb) / len(hb): print('FAIL: headline fell'); fail = 1
sys.exit(1 if fail else 0)

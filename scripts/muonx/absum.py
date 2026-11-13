#!/usr/bin/env python3
"""Muon A/B summary: per arm the mean training loss (and per-head losses) over the last 300 logged steps, paired against the
reference arm on the same steps (--det 1: the same windows), with a block-bootstrap 95% CI of the mean difference (blocks of 5 logged
steps, autocorrelation), and the mean samp/s. usage: absum.py [REF=muon] [LAST=300]"""
import re, sys, os, random, glob
D = os.path.dirname(os.path.abspath(__file__)); ref = sys.argv[1] if len(sys.argv) > 1 else 'muon'; last = int(sys.argv[2]) if len(sys.argv) > 2 else 300
KEYS = [('loss', r' loss ([0-9.]+)'), ('phase', r'phase bce ([0-9.]+)'), ('rv', r'rv bce ([0-9.]+)'), ('normal', r'normal bce ([0-9.]+)'),
        ('wind4', r'wind4 bce ([0-9.]+)'), ('ink', r'ink bce ([0-9.]+)'), ('mil', r'ink mil ([0-9.]+)'), ('fib', r'fib[a-z ]* ([0-9.]+)')]
def load(p):
    rows = {}; sps = []; done = False
    for l in open(p, errors='replace'):
        m = re.match(r'step\s+(\d+) ', l)
        if not m: continue
        r = {}
        for k, rx in KEYS:
            x = re.search(rx, l)
            if x: r[k] = float(x.group(1))
        x = re.search(r'([0-9.]+) samp/s', l)
        if x: sps.append(float(x.group(1)))
        rows[int(m.group(1))] = r
    return rows, sps
arms = {os.path.basename(p)[:-4]: load(p) for p in sorted(glob.glob(f'{D}/runs/*.log'))}
if ref not in arms: sys.exit(f'no reference arm {ref}')
R = arms[ref][0]; smax = max(R); lo = smax - last
print(f'reference {ref}: steps {lo + 1}..{smax}; per arm the mean, and arm - {ref} on the same steps [95% block-bootstrap CI]')
hdr = ['arm', 'steps', 'samp/s'] + [k for k, _ in KEYS]; print('\t'.join(hdr))
random.seed(1)
for a, (rows, sps) in arms.items():
    st = sorted(s for s in rows if s > lo and s in R)
    if not st: print(f'{a}\t{max(rows) if rows else 0}\t(running)'); continue
    out = [a, str(max(rows)), f'{sum(sps[-30:]) / max(1, len(sps[-30:])):.3f}']
    for k, _ in KEYS:
        d = [rows[s][k] - R[s][k] for s in st if k in rows[s] and k in R[s]]
        v = [rows[s][k] for s in st if k in rows[s]]
        if not v: out.append('-'); continue
        if a == ref or not d: out.append(f'{sum(v) / len(v):.4f}'); continue
        B = [d[i:i + 5] for i in range(0, len(d), 5)]; bs = []
        for _ in range(2000):
            x = [y for _ in B for y in random.choice(B)]; bs.append(sum(x) / len(x))
        bs.sort(); out.append(f'{sum(v) / len(v):.4f} {sum(d) / len(d):+.4f} [{bs[50]:+.4f},{bs[1949]:+.4f}]')
    print('\t'.join(out))

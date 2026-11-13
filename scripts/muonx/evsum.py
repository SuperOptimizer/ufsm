#!/usr/bin/env python3
"""Held-out val of the finished arms (runs/evNAME.log: resume, one lr-0 step, the same 32 val windows, live weights) against the reference
(default muon): total val loss, phase / normal / wind4 bce, turn acc, ink bce; plus the arm's wall time (done in ... s). usage: evsum.py [REF]"""
import re, os, glob, sys
D = os.path.dirname(os.path.abspath(__file__)); ref = sys.argv[1] if len(sys.argv) > 1 else 'muon'
def ev(p):
    t = open(p, errors='replace').read(); r = {}
    m = re.search(r'val unwrap2: phase bce ([0-9.]+) normal bce ([0-9.]+) wind4 bce ([0-9.]+).*?turn acc ([0-9.]+)', t)
    if m: r.update(phase=float(m[1]), normal=float(m[2]), wind4=float(m[3]), acc=float(m[4]))
    m = re.search(r'val ink: bce ([0-9.]+)', t); r['ink'] = float(m[1]) if m else None
    m = re.findall(r' val ([0-9.]+),', t); r['val'] = float(m[-1]) if m else None
    return r
def wall(a):
    p = f'{D}/runs/{a}.log'
    m = re.search(r'done in (\d+)s', open(p, errors='replace').read()) if os.path.exists(p) else None
    return int(m[1]) if m else None
E = {os.path.basename(p)[2:-4]: ev(p) for p in sorted(glob.glob(f'{D}/runs/ev*.log'))}
E = {a: r for a, r in E.items() if r.get('val') is not None}
if ref not in E: sys.exit(f'no eval of {ref}')
R = E[ref]; wr = wall(ref)
print('arm\twall s\tval\tphase\tnormal\twind4\tturn acc\tink   (arm, and arm - ' + ref + ')')
for a, r in E.items():
    w = wall(a); row = [a, f'{w} ({w / wr:.2f}x)' if w and wr else '-']
    for k in ('val', 'phase', 'normal', 'wind4', 'acc', 'ink'):
        v = r.get(k); row.append('-' if v is None else f'{v:.4f}' if a == ref else f'{v:.4f} {v - R[k]:+.4f}')
    print('\t'.join(row))

#!/usr/bin/env python3
"""scripts/pseudo/dirmask.py CELLS.tsv OUT.boxes [--c 128 (or the cells header's chunk edge)] [--minal 20000]: `ufsm phase-check --cells` rows (cz cy cx pairs up aligned
aligned_up, chunk units of C voxels) -> phase-coh --box-scale lines "z y x nz ny nx 0" for the chunks the aligned test reads reversed:
>= MINAL aligned pairs (~1 % of a 128^3 chunk: a coherent patch, binomial sd < 0.004) of which < half rise outward (scripts/univ/
b3_1299_mask.py rule (b); the raw share is confounded with sheet orientation, DESIGN 2026-11-12 "PHerc1299 B3 labels"). Prints the shares."""
import sys
a = sys.argv[1:]
def arg(k, d):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return int(v)
    return d
C, MINAL = arg('--c', 128), arg('--minal', 20000)
src, out = a
n = al = alup = rev = revp = cells = 0
with open(out, 'w') as fo:
    for l in open(src):
        if l.startswith('# chunk') and '(x ' in l: C = int(l.split('(x ')[1].split()[0])   # (the chunk edge phase-check used)
        if l[0] == '#' or not l.strip() or not l.split()[0].lstrip('-').isdigit(): continue
        cz, cy, cx, pr, up, an, aup = map(int, l.split()[:7]); cells += 1; n += pr; al += an; alup += aup
        if an >= MINAL and aup < 0.5 * an:
            rev += 1; revp += pr; fo.write('%d %d %d %d %d %d 0\n' % (cz * C, cy * C, cx * C, C, C, C))
print('dirmask: %d chunks, aligned outward %.4f, %d reversed chunks masked (%.4f of the pairs)' % (cells, alup / max(al, 1), rev, revp / max(n, 1)))

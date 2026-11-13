#!/usr/bin/env python3
"""scripts/univ/b3_1299_mask.py CELLS.tsv FIBRE.boxes OUT.boxes [--gated CELLS_GATED.tsv]
The PHerc1299 phase-weight boxes (coordinator's option B, 2026-11-12) for `ufsm phase-coh --coh 0 --box-scale OUT.boxes`, on the
level-2 (9.596 um) phase grid; the first box holding a voxel wins, so the order is:
  1. the held-out band + guard (level-2 z < 4167 = level-0 z < 16668, holdouts-1299.json "1299-zband"): 0
  2. the direction-masked 128^3 cells: 0. A cell is masked when
       (a) its fibre tile has factor 0 AND its raw outward share is < 0.8 (the two checks agree), or
       (b) the aligned-voxel test marks it reversed: >= MINAL (20000) aligned pairs (|cos| of the phase gradient to the radius > 0.7,
           ufsm phase-check --cells) of which < 0.5 rise outward. 20000 = ~1% of a cell's voxels: a coherent patch, not speckle
           (the binomial sd of the share is then < 0.004).
  3. the fibre tiles (spirals' fiber-judge boxes, Paris 4 ramp 189..378 /m): 0.5 + 0.5 x factor. Floored at 0.5 because the ramp is
     miscalibrated on this scan (it zeroes 54% of the pairs; median rate ~400 /m everywhere) and must not act as a mask.
Everything else: 1 (--box-default 1). The papyrus mask and the gate come from phase-coh itself (--papyrus, run on the gated phase).
CELLS.tsv: phase-check --cells of the labelled phase (cz cy cx pairs up aligned aligned_up). Prints the shares of voxel pairs."""
import sys
a = sys.argv[1:]; gated = None
if '--gated' in a: i = a.index('--gated'); gated = a[i + 1]; del a[i:i + 2]
cells_p, fib_p, out_p = a
C = 128; BAND = 4167; MINAL = 20000; LOW = 0.8
T = []
for l in open(fib_p):
    if l[0] == '#' or not l.strip(): continue
    v = l.split(); T.append(([int(x) for x in v[:6]], float(v[6])))
def tile(z, y, x):
    for b, s in T:
        if b[0] <= z < b[0] + b[3] and b[1] <= y < b[1] + b[4] and b[2] <= x < b[2] + b[5]: return s
    return None
def rows(p):
    for l in open(p):
        if l[0] == '#': continue
        cz, cy, cx, n, up, an, aup = map(int, l.split()); yield cz, cy, cx, n, up, an, aup
masked = []; st = dict(all=0, band=0, a=0, b=0, both=0, out=0, kept_w=0.0, al=0, alup=0, up=0, lowraw=0, rev_cells=0, a_cells=0, cells=0)
for cz, cy, cx, n, up, an, aup in rows(cells_p):
    st['all'] += n
    if cz * C < BAND: st['band'] += n; continue      # (cells straddling the band edge: the band box masks their lower part voxel-exactly)
    st['cells'] += 1; st['out'] += n; st['al'] += an; st['alup'] += aup; st['up'] += up
    f = tile(cz * C + C // 2, cy * C + C // 2, cx * C + C // 2); sc = up / n
    ra = f == 0 and sc < LOW; rb = an >= MINAL and aup < 0.5 * an
    if sc < LOW: st['lowraw'] += n
    if ra: st['a'] += n; st['a_cells'] += 1
    if rb: st['b'] += n; st['rev_cells'] += 1
    if ra and rb: st['both'] += n
    if ra or rb: masked.append((cz, cy, cx)); continue
    st['kept_w'] += n * (1.0 if f is None else 0.5 + 0.5 * f)
with open(out_p, 'w') as F:
    F.write(f'# b3_1299_mask.py: band z < {BAND}; {len(masked)} direction-masked {C}^3 cells (fibre 0 and outward < {LOW}, or >= {MINAL} aligned pairs with < 0.5 outward); fibre tiles 0.5 + 0.5 x factor\n# z y x nz ny nx s\n')
    F.write(f'0 0 0 {BAND} 1000000 1000000 0\n')
    for cz, cy, cx in masked: F.write(f'{cz * C} {cy * C} {cx * C} {C} {C} {C} 0\n')
    for b, s in T: F.write(f'{b[0]} {b[1]} {b[2]} {b[3]} {b[4]} {b[5]} {0.5 + 0.5 * s:.4f}\n')
o = st['out']
print(f"cells scored {st['cells']} outside the band (pairs: {st['all']} all, {st['band'] / st['all']:.4f} in the band cells)")
print(f"outside the band: raw outward {st['up'] / o:.4f}; aligned pairs {st['al'] / o:.4f} of the pairs, aligned outward {st['alup'] / max(1, st['al']):.4f}")
print(f"masked (a) fibre 0 and outward < {LOW}: {st['a_cells']} cells, {st['a'] / o:.4f} of the pairs; (b) reversed by the aligned test (>= {MINAL} aligned, < 0.5 outward): {st['rev_cells']} cells, {st['b'] / o:.4f}; both {st['both'] / o:.4f}")
print(f"direction-masked total {len(masked)} cells, {(st['a'] + st['b'] - st['both']) / o:.4f} of the pairs; kept {1 - (st['a'] + st['b'] - st['both']) / o:.4f}; mean phase weight over all pairs outside the band (fibre scale, before papyrus / gate) {st['kept_w'] / o:.4f}")
if gated:
    key = set(masked); g = dict(all=0, band=0, out=0, m=0, up=0, al=0, alup=0)
    for cz, cy, cx, n, up, an, aup in rows(gated):
        g['all'] += n
        if cz * C < BAND: g['band'] += n; continue
        g['out'] += n; g['up'] += up; g['al'] += an; g['alup'] += aup
        if (cz, cy, cx) in key: g['m'] += n
    print(f"gated phase: pairs outside the band {g['out']} = {g['out'] / o:.4f} of the labelled phase's (the gate + shard rule keep that share; {1 - g['out'] / o:.4f} removed), "
          f"raw outward {g['up'] / g['out']:.4f}, aligned outward {g['alup'] / max(1, g['al']):.4f}; direction-masked {g['m'] / g['out']:.4f} of them; band cells held {g['band'] / g['all']:.4f} of the gated pairs")

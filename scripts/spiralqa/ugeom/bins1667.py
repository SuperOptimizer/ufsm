#!/usr/bin/env python3
"""audit-ugeom: tables of the pub1667 comparison binned by z / radius / pitch / hole distance / angle / our wrap, handedness ramp removed."""
import sys
import numpy as np
npz, aux = sys.argv[1:3]
Z = np.load(npz); X = np.load(aux)
d = Z['d']; s = Z['s']; off = Z['off'].astype(np.int32); A = Z['area']; nearcut = Z['nearcut']; jid = Z['id'].astype(np.int32); T = Z['T']
rad = X['rad']; pitch = X['pitch']; hole = X['hole']; z = X['z']; u = X['u']
close = d < 100
g = close & ~nearcut
# handedness ramp: per u bin (64) the mode of off
NB = 64
ub = np.minimum((u * NB).astype(np.int32), NB - 1)
ramp = np.zeros(NB, np.int32); rshare = np.zeros(NB)
for b in range(NB):
    m = g & (ub == b)
    v, c = np.unique(off[m], return_counts=True)
    ramp[b] = v[np.argmax(c)]; rshare[b] = c.max() / c.sum()
print('ramp by u (64 bins):', ''.join(str(r) for r in ramp))
print('mode share per u bin: min %.2f median %.2f' % (rshare.min(), np.median(rshare)))
# continuous fit: off ~ k*u + c, best k
best = None
for k in np.arange(-3, 3.01, 0.25):
    for c in np.arange(-1, 1.01, 0.125):
        sh = (off[g][::50] == np.round(k * u[g][::50] + c)).mean()
        if best is None or sh > best[0]: best = (sh, k, c)
print('best linear ramp off = round(%.2f u + %.3f): share %.3f' % (best[1], best[2], best[0]))
e = off - ramp[ub]
same_raw = (off == 0)
print('identity same (raw off==0) %.3f; after the ramp %.3f; e=+1 %.3f e=-1 %.3f |e|>=2 %.3f' % (
    same_raw[g].mean(), (e[g] == 0).mean(), (e[g] == 1).mean(), (e[g] == -1).mean(), (np.abs(e[g]) >= 2).mean()))

def table(name, key, edges, fmt='%g'):
    print(f'\n== by {name}')
    print(f'{name:>14} area%  cov50  cov100  d_med  s_med  pitch_med  same_raw  same_ramp  e+-1  |e|>=2')
    tot = A.sum()
    for i in range(len(edges) - 1):
        m = (key >= edges[i]) & (key < edges[i + 1])
        if m.sum() < 1000: continue
        a = A[m]; mg = m & g
        lab = (fmt % edges[i]) + '-' + (fmt % edges[i + 1])
        print(f'{lab:>14} {a.sum()/tot*100:5.1f} {a[d[m]<50].sum()/a.sum():6.3f} {a[d[m]<100].sum()/a.sum():6.3f} {np.median(d[m]):6.0f} {np.median(s[m&close]) if (m&close).any() else np.nan:6.0f} '
              f'{np.nanmedian(pitch[m]):9.0f} {same_raw[mg].mean():9.3f} {(e[mg]==0).mean():10.3f} {(np.abs(e[mg])==1).mean():5.3f} {(np.abs(e[mg])>=2).mean():6.3f}')

table('z_L0', z, [0, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000])
table('radius_mm', rad, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
table('pitch_um', pitch, [-1e9, 0, 100, 150, 200, 250, 300, 400, 500, 700, 1e9])
pn = np.where(np.isfinite(pitch), 0, 1)
m = ~np.isfinite(pitch); print('\npitch unknown (outermost turn / hole): area %.3f cov50 %.3f' % (A[m].sum() / A.sum(), A[m & (d < 50)].sum() / A[m].sum()))
table('hole_steps', hole, [0, 2, 4, 8, 16, 32, 64, 1e9])
table('u_angle', u, list(np.arange(0, 1.0001, 0.0625)), '%.3f')
table('our_id', jid.astype(np.float32), [-20, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 63], '%d')
table('pub_turn', T, list(range(0, 33, 2)), '%d')

#!/usr/bin/env python3
"""audit-ugeom: bin the pub1667 comparison (gt/pub1667-cmp/full/pub_dist.npz, surfcmp.py) by z, radius, angle, local pitch,
distance to a hole of the published surface, and our wrap id; separate the handedness ramp from the residual sheet identity.
  budget1667.py PUB_TIFXYZ NPZ AXIS.json OUT_PREFIX"""
import sys, json
import numpy as np, tifffile
from scipy import ndimage

pub, npz, axf, out = sys.argv[1:5]
um = 2.399
Z = np.load(npz)
H, Wd = (int(v) for v in Z['shape'])
vi = Z['vi']; d = Z['d']; s = Z['s']; jid = Z['id'].astype(np.int32); T = Z['T']; off = Z['off'].astype(np.int32); npub = Z['npub'].astype(np.int32)
nearcut = Z['nearcut']; A = Z['area']
print('vertices', len(vi), flush=True)
z = tifffile.imread(f'{pub}/z.tif'); valid = (z > 0) & np.isfinite(z)
zz = z.ravel()[vi].astype(np.float32)
y = tifffile.imread(f'{pub}/y.tif'); x = tifffile.imread(f'{pub}/x.tif')
valid &= (y > 0) & (x > 0)
ax = json.load(open(axf)); ax = ax['control_points'] if isinstance(ax, dict) else ax
a = np.array(sorted((p['z'], p['y'], p['x']) for p in ax))
cyg = np.interp(z, a[:, 0], a[:, 1]).astype(np.float32); cxg = np.interp(z, a[:, 0], a[:, 2]).astype(np.float32)
radg = np.hypot(y - cyg, x - cxg).astype(np.float32); radg[~valid] = np.nan
del cyg, cxg
rad = radg.ravel()[vi] * um / 1000      # mm
# continuous turn on the grid (rebuild from npz T at vi)
Tg = np.full(H * Wd, np.nan, np.float32); Tg[vi] = T; Tg = Tg.reshape(H, Wd)
# local pitch: along each row, the radius one turn further out (T + 1) minus here
pitch = np.full((H, Wd), np.nan, np.float32)
for r in range(H):
    m = np.isfinite(Tg[r])
    if m.sum() < 100: continue
    c = np.flatnonzero(m); t = Tg[r, c]; rr = radg[r, c]
    o = np.argsort(t, kind='stable'); t = t[o]; c = c[o]; rr = rr[o]
    j = np.searchsorted(t, t + 1.0)
    ok = j < len(t)
    jj = np.minimum(j, len(t) - 1)
    good = ok & (np.abs(t[jj] - t - 1.0) < 0.01)
    p = np.where(good, rr[jj] - rr, np.nan)
    pitch[r, c] = p
pv = pitch.ravel()[vi] * um     # um
del pitch
# distance to a hole / the edge of the published surface (grid steps; one column step ~52 um, one row step = 20 voxels?)
hole = ndimage.distance_transform_edt(valid).astype(np.float32)
hv = hole.ravel()[vi]
del hole
row = (vi // Wd).astype(np.int32); col = (vi % Wd).astype(np.int32)
# angle fraction u from our cut, as surfcmp: npub = round(T - u - c0) -> u = frac
c0 = -0.016265556385455595
u = np.mod(T - npub - c0, 1.0)
np.savez(out + '_aux.npz', rad=rad.astype(np.float32), pitch=pv.astype(np.float32), hole=hv, z=zz, u=u.astype(np.float32), row=row, col=col)
print('aux written', flush=True)

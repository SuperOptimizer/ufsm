#!/usr/bin/env python3
# axcheck.py RAW NZ NY NX F UM0 AXIS OUTJSON: the corpus axis vs the papyrus centroid on a coarse level (F = level-0 voxels per coarse
# voxel, UM0 = the level-0 voxel um). Papyrus = voxels above an Otsu threshold of the nonzero CT; centroid per z, running median +-2048
# level-0 voxels. Prints the offsets at 5 z levels (10..90 % of the papyrus z extent) in mm and whether the axis lies in the papyrus
# (any papyrus within 1 mm). Writes OUTJSON (the centroid axis, control points every 32 level-0 voxels over the axis file's z range).
# Exit 0: axis ok; 3: off (median of the 30/50/70 % offsets > 2 mm, or no papyrus within 1 mm at 50 %) -> the caller uses OUTJSON.
import sys, json, numpy as np
raw, nz, ny, nx, F, um0, axf, out = sys.argv[1], *map(int, sys.argv[2:5]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8]
a = np.fromfile(raw, np.uint8).reshape(nz, ny, nx)
v = a[::4][a[::4] > 0]; h = np.bincount(v, minlength=256).astype(float); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
sb = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12); th = int(np.argmax(sb))
Z = []; Y = []; X = []; M = {}
for z in range(nz):
    m = a[z] > th
    if m.sum() < 50: continue
    yy, xx = np.nonzero(m); Z.append((z + .5) * F); Y.append((yy.mean() + .5) * F); X.append((xx.mean() + .5) * F); M[z] = m
Z = np.array(Z); Y = np.array(Y); X = np.array(X)
r = max(1, int(2048 / F))
med = lambda q: np.array([np.median(q[max(0, i - r):i + r + 1]) for i in range(len(q))])
Ys, Xs = med(Y), med(X)
ax = json.load(open(axf))['control_points']; az = np.array([q['z'] for q in ax]); ay = np.array([q['y'] for q in ax]); axx = np.array([q['x'] for q in ax])
bad = 0; rows = []; mid = []; in50 = True; rpx = max(1, int(1000 / (um0 * F)))
for f in (.1, .3, .5, .7, .9):
    z0 = Z[0] + f * (Z[-1] - Z[0]); i = int(np.argmin(abs(Z - z0)))
    oy, ox = np.interp(Z[i], az, ay), np.interp(Z[i], az, axx)
    d = np.hypot(oy - Ys[i], ox - Xs[i]) * um0 / 1000
    m = M[int(Z[i] / F)]; cy, cx = int(oy / F), int(ox / F)
    inside = 0 <= cy < ny and 0 <= cx < nx and m[max(0, cy - rpx):cy + rpx + 1, max(0, cx - rpx):cx + rpx + 1].any()
    rows.append(f"z {int(Z[i])}: {d:.2f} mm{'' if inside else ' OUTSIDE'}")
    if f in (.3, .5, .7): mid.append(d)
    if f == .5: in50 = inside
bad = float(np.median(mid)) > 2 or not in50   # (lead 11-12: the centroid is not the umbilicus; the ends are tapered caps)
zz = np.arange(az[0], az[-1] + 1, 32.0)
json.dump({"control_points": [{"z": float(z), "y": float(np.interp(z, Z, Ys)), "x": float(np.interp(z, Z, Xs))} for z in zz],
           "source": f"papyrus centroid (coarse x{F:g}, Otsu > {th}, running median +-2048 L0 voxels)"}, open(out, "w"))
print(f"axcheck: threshold {th}; " + "; ".join(rows) + (f"  median30-70 {np.median(mid):.2f} mm") + ("  -> OFF: use the centroid axis" if bad else "  -> ok"))
sys.exit(3 if bad else 0)

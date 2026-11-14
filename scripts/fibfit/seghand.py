#!/usr/bin/env python3
"""seghand.py TIFXYZ_DIR [...]: the spiral's hand from published segments (tifxyz x / y / z, any voxel frame, axis not needed).
Per grid row: the valid vertices in column order, theta about the row's ring centroid, unwrapped along the columns. A row that covers
>= 0.8 turn votes the sign of (r at its last 5 % columns - r at its first 5 %) x sign(total dtheta): the radius grows with the unwrapped
angle (dir +1, src/fit.h) or against it (dir -1). The sheet's non-circular shape is continuous in theta and nearly cancels over one
turn; the spiral's own step is one pitch per turn. Rows covering > 1.2 turns use the per-turn radial step (r(theta + 2 pi) - r(theta)).
Prints per segment: rows voting +1 / -1 and the median signed step (voxels)."""
import sys, numpy as np, tifffile
def hand(d, every=8):
    X = tifffile.imread(f'{d}/x.tif').astype(np.float64); Y = tifffile.imread(f'{d}/y.tif').astype(np.float64); Z = tifffile.imread(f'{d}/z.tif')
    V = (Z > 0) & (X > 0) & (Y > 0) & np.isfinite(X)
    votes = []
    for i in range(0, X.shape[0], every):
        m = V[i]
        if m.sum() < 50: continue
        x, y = X[i, m], Y[i, m]
        cx, cy = x.mean(), y.mean()
        th = np.unwrap(np.arctan2(y - cy, x - cx)); r = np.hypot(y - cy, x - cx); tot = th[-1] - th[0]
        if abs(tot) < 0.8 * 2 * np.pi: continue
        s = np.sign(tot)
        if abs(tot) > 1.2 * 2 * np.pi:   # several turns: r one turn later at the same angle
            a = th * s; j = np.searchsorted(a, a + 2 * np.pi); ok = j < len(a)
            step = np.median(r[j[ok]] - r[ok]); votes.append(step if s > 0 else -step)
        else:
            k = max(2, len(r) // 20); votes.append(s * (np.median(r[-k:]) - np.median(r[:k])))
    v = np.array(votes)
    return (v > 0).sum(), (v < 0).sum(), float(np.median(v)) if len(v) else float('nan')
if __name__ == '__main__':
    for d in sys.argv[1:]:
        p, n, md = hand(d)
        print(f'{d}\trows +1 {p}\t-1 {n}\tmedian step {md:+.1f}\tdir {"+1" if p > 2 * n else "-1" if n > 2 * p else "?"}', flush=True)

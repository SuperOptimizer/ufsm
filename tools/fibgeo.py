"""fibgeo.py AFV CT.raw --box z,y,x,nz,ny,nx --level L [--tr T.json] [--shift R] [--fam H|V]: do AFV fibre centrelines sit on bright CT
ridges of OUR volume? (analysis only; python3 -I). CT.raw: uint8 box from `ufsm read` (level-L voxels of our scan). AFV native xyz ->
our level-L zyx: optional inverse of the published transform (ours L0 xyz -> AFV xyz), then (v + 0.5) / 2^L - 0.5.
Prints: the across-fibre CT profile (ring mean at radius r about the centreline), the 3-D translation search (mean CT along the
centrelines moved by (dz, dy, dx)), per family."""
import argparse, json, sqlite3, sys
import numpy as np
from scipy.ndimage import map_coordinates

ap = argparse.ArgumentParser()
ap.add_argument("afv"); ap.add_argument("ct"); ap.add_argument("--box", required=True); ap.add_argument("--level", type=int, default=0)
ap.add_argument("--tr"); ap.add_argument("--shift", type=int, default=6); ap.add_argument("--step", type=float, default=1.0)
ap.add_argument("--half", type=int, default=0, help="1: coordinates are v / 2^L (no half-voxel centre shift)")
a = ap.parse_args()
bz, by, bx, nz, ny, nx = map(int, a.box.split(","))
ct = np.fromfile(a.ct, dtype=np.uint8).reshape(nz, ny, nx).astype(np.float32)
sc = 2.0 ** a.level
M = None
if a.tr:
    M = np.array(json.load(open(a.tr))["transformation_matrix"], dtype=np.float64)[:3]
    Mi = np.linalg.inv(M[:, :3])

def to_level(xyz):   # AFV native xyz (n,3) -> our level zyx
    if M is not None: xyz = (xyz - M[:, 3]) @ Mi.T
    v = xyz / sc if a.half else (xyz + 0.5) / sc - 0.5
    return v[:, ::-1]

def from_level_corners():   # the box corners in AFV xyz (for the rtree query)
    c = np.array([[bz + i * nz, by + j * ny, bx + k * nx] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)
    xyz = (c[:, ::-1] + 0.5) * sc - 0.5
    if M is not None: xyz = xyz @ M[:, :3].T + M[:, 3]
    return xyz.min(0), xyz.max(0)

lo, hi = from_level_corners()
db = sqlite3.connect(f"file:{a.afv}?mode=ro&immutable=1", uri=True)
q = ("select b.fiber_id, b.first_segment, b.points, f.family from block_bounds r join blocks b on b.id = r.id join fibers f on f.id = b.fiber_id "
     "where r.max_x >= ? and r.min_x <= ? and r.max_y >= ? and r.min_y <= ? and r.max_z >= ? and r.min_z <= ? order by b.fiber_id, b.first_segment")
runs = []; pf = ps = None
for fid, seg, blob, fam in db.execute(q, (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])):
    p = np.frombuffer(blob, dtype="<f8").reshape(-1, 3)
    if runs and fid == pf and seg == ps + 256: runs[-1][1] = np.vstack([runs[-1][1], p[1:]])
    else: runs.append([fam, p])
    pf, ps = fid, seg
P = []; T = []; F = []
for fam, p in runs:
    v = to_level(p); d = np.linalg.norm(np.diff(v, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 2: continue
    t = np.arange(0, s[-1], a.step); r = np.stack([np.interp(t, s, v[:, k]) for k in range(3)], 1)
    tg = np.gradient(r, axis=0); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
    P.append(r); T.append(tg); F.append(np.full(len(r), {"H": 0, "V": 1}.get(fam, 2)))
P = np.vstack(P) - [bz, by, bx]; T = np.vstack(T); F = np.concatenate(F)
m = np.all((P >= 12) & (P < np.array([nz, ny, nx]) - 13), axis=1); P, T, F = P[m], T[m], F[m]
print(f"runs {len(runs)}, points in box {len(P)} ({len(P) * a.step * sc * 1e-3:.0f} level-0 kvox of arc)  H {np.sum(F == 0)} V {np.sum(F == 1)} ? {np.sum(F == 2)}")
samp = lambda Q: map_coordinates(ct, Q.T, order=1, mode="nearest")
allm = ct[ct > 0]; print(f"CT box: mean {ct.mean():.1f}, nonzero share {np.mean(ct > 0):.3f}, nonzero mean {allm.mean():.1f} sd {allm.std():.1f}")
# perpendicular frame
ref = np.where(np.abs(T[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]); e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(T, e1)
for name, sel in (("all", F >= 0), ("H", F == 0), ("V", F == 1), ("?", F == 2)):
    if sel.sum() < 100: continue
    out = []
    for r in (0, 1, 2, 3, 4, 6, 8, 12):
        if r == 0: out.append(samp(P[sel]).mean()); continue
        acc = 0
        for k in range(8):
            ang = k * np.pi / 4; acc += samp(P[sel] + r * (np.cos(ang) * e1[sel] + np.sin(ang) * e2[sel])).mean()
        out.append(acc / 8)
    print(f"  {name:>3} ring profile r=0,1,2,3,4,6,8,12 vox:", " ".join(f"{v:.1f}" for v in out))
    # tangent z-ness: share of points with |t_z| > 0.7 (axial fibres)
    print(f"      |t_z| > 0.7: {np.mean(np.abs(T[sel][:, 0]) > 0.7):.3f}   |t_z| < 0.3: {np.mean(np.abs(T[sel][:, 0]) < 0.3):.3f}")
R = a.shift; sub = P[:: max(1, len(P) // 200000)]
best = (-1, None); grid = np.zeros((2 * R + 1,) * 3)
for dz in range(-R, R + 1):
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            v = samp(sub + [dz, dy, dx]).mean(); grid[dz + R, dy + R, dx + R] = v
            if v > best[0]: best = (v, (dz, dy, dx))
print(f"translation search +-{R}: at 0 {grid[R, R, R]:.2f}, best {best[0]:.2f} at dz,dy,dx {best[1]}, grid mean {grid.mean():.2f} min {grid.min():.2f}")
for ax, nm in enumerate("zyx"):
    idx = [R, R, R]; line = []
    for k in range(2 * R + 1): idx[ax] = k; line.append(grid[tuple(idx)])
    print(f"   along {nm}:", " ".join(f"{v:.1f}" for v in line))

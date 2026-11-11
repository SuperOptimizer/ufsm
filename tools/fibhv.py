"""fibhv.py AFV --box z,y,x,nz,ny,nx --level L --axis AXIS.json [--tr T] [--ct CT.raw]: H / V fibre layer order along the radius
(analysis; python3 -I). For H-fibre points, the nearest V-fibre point within --rad voxels: signed radial offset (V - H) . r_hat, r_hat
pointing away from the axis (umbilicus). Recto = H layer facing the umbilicus would give V outward (positive). With --ct, pairs are
kept only when the CT along the H -> V segment stays above the box's Otsu threshold (same sheet: no gap between them)."""
import argparse, json, sys
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from fiblib import Map, open_afv, runs_in_box
ap = argparse.ArgumentParser(); ap.add_argument("afv"); ap.add_argument("--box", required=True); ap.add_argument("--level", type=int, default=0); ap.add_argument("--axis", required=True)
ap.add_argument("--tr"); ap.add_argument("--rad", type=float, default=10); ap.add_argument("--ct"); a = ap.parse_args()
box = tuple(map(int, a.box.split(","))); bz, by, bx, nz, ny, nx = box; sc = 2.0 ** a.level
cp = json.load(open(a.axis))["control_points"]; cp.sort(key=lambda c: c["z"]); az = np.array([c["z"] for c in cp]) / sc; ay = np.array([c["y"] for c in cp]) / sc; ax = np.array([c["x"] for c in cp]) / sc
mp = Map(a.level, a.tr); db, vs = open_afv(a.afv); runs = runs_in_box(db, mp, box, step=2.0)
P = [[], []]
for fid, fam, flen, r, tg in runs:
    m = np.all((r >= [bz, by, bx]) & (r < [bz + nz, by + ny, bx + nx]), axis=1)
    if fam < 2 and m.any(): P[fam].append(r[m])
H = np.vstack(P[0]); V = np.vstack(P[1]); t = cKDTree(V); d, i = t.query(H, distance_upper_bound=a.rad); ok = np.isfinite(d); H = H[ok]; Vn = V[i[ok]]; d = d[ok]
rh = np.stack([np.zeros(len(H)), H[:, 1] - np.interp(H[:, 0], az, ay), H[:, 2] - np.interp(H[:, 0], az, ax)], 1); rad = np.linalg.norm(rh, axis=1); rh /= rad[:, None]
off = np.sum((Vn - H) * rh, axis=1)
keep = np.ones(len(H), bool)
if a.ct:
    ct = np.fromfile(a.ct, dtype=np.uint8).reshape(nz, ny, nx).astype(np.float32)
    h = np.bincount(ct.astype(np.uint8).ravel(), minlength=256).astype(float); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"): thr = int(np.nanargmax((mu[-1] * w - mu) ** 2 / (w * (1 - w))))
    mn = np.full(len(H), 255.0)
    for s in np.linspace(0, 1, 9):
        q = H + s * (Vn - H) - [bz, by, bx]; mn = np.minimum(mn, map_coordinates(ct, q.T, order=1, mode="nearest"))
    keep = mn > thr; print(f"CT Otsu {thr}: pairs joined by papyrus {keep.mean():.3f}")
print(f"H points {len(ok)}, with a V point within {a.rad:g} voxels {ok.mean():.3f}; mean radius {rad.mean():.0f} voxels")
for name, m in (("all pairs", np.ones(len(H), bool)), ("joined by papyrus", keep), ("joined, |offset| >= 2", keep & (np.abs(off) >= 2)), ("joined, distance <= 6", keep & (d <= 6))):
    if m.sum() < 50: continue
    hh = np.histogram(off[m], bins=10, range=(-10, 10))[0] / m.sum()
    print(f"  {name:24s} n {m.sum():7d}: V outward of H {np.mean(off[m] > 0):.3f}, mean offset {off[m].mean():+.2f}, median {np.median(off[m]):+.2f} | hist -10..10 step 2: " + " ".join(f"{x:.3f}" for x in hh))

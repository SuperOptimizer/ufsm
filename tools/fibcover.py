"""fibcover.py AFV CT.raw --box z,y,x,nz,ny,nx --level L [--tr T] --um U: fibre density and coverage in one CT box (analysis; python3 -I).
Papyrus = CT above the Otsu threshold of the box. Prints fibres (distinct ids) and metres per mm^3 of box / of papyrus, the share of
papyrus voxels within 1 / 2 / 4 / 8 voxels of a centreline, and the share of centreline points on papyrus."""
import argparse, sys
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
import numpy as np
from scipy.ndimage import distance_transform_edt
from fiblib import Map, open_afv, runs_in_box
ap = argparse.ArgumentParser(); ap.add_argument("afv"); ap.add_argument("ct"); ap.add_argument("--box", required=True); ap.add_argument("--level", type=int, default=0)
ap.add_argument("--tr"); ap.add_argument("--um", type=float, required=True); a = ap.parse_args()
box = tuple(map(int, a.box.split(","))); bz, by, bx, nz, ny, nx = box
ct = np.fromfile(a.ct, dtype=np.uint8).reshape(nz, ny, nx)
h = np.bincount(ct.ravel(), minlength=256).astype(float); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
with np.errstate(divide="ignore", invalid="ignore"): sb = (mt * w - mu) ** 2 / (w * (1 - w))
thr = int(np.nanargmax(sb)); pap = ct > thr
mp = Map(a.level, a.tr); db, vs = open_afv(a.afv); runs = runs_in_box(db, mp, box, step=0.5)
cl = np.zeros(ct.shape, bool); arc = [0.0, 0.0, 0.0]; ids = [set(), set(), set()]; onp = 0; npt = 0
for fid, fam, flen, r, tg in runs:
    q = np.rint(r - [bz, by, bx]).astype(int); m = np.all((q >= 0) & (q < [nz, ny, nx]), axis=1)
    if not m.any(): continue
    cl[q[m, 0], q[m, 1], q[m, 2]] = True; arc[fam] += m.sum() * 0.5; ids[fam].add(fid); onp += pap[q[m, 0], q[m, 1], q[m, 2]].sum(); npt += m.sum()
vmm3 = (a.um * 1e-3) ** 3; V = ct.size * vmm3; Vp = pap.sum() * vmm3; L = sum(arc) * a.um * 1e-6; N = sum(len(s) for s in ids)
print(f"box {V:.1f} mm^3, Otsu {thr}, papyrus share {pap.mean():.3f} ({Vp:.1f} mm^3); fibres {N} (H {len(ids[0])} V {len(ids[1])} ? {len(ids[2])}), arc {L:.2f} m (H {arc[0] * a.um * 1e-6:.2f} V {arc[1] * a.um * 1e-6:.2f})")
print(f"  per mm^3 of box: {N / V:.0f} fibres, {L / V:.3f} m ; per mm^3 of papyrus: {N / Vp:.0f} fibres, {L / Vp:.3f} m ; centreline points on papyrus {onp / max(1, npt):.3f}")
d = distance_transform_edt(~cl)
print("  papyrus voxels within r of a centreline: " + "  ".join(f"r<={r}: {np.mean(d[pap] <= r):.3f}" for r in (1, 2, 4, 8)) + f"   median distance {np.median(d[pap]):.1f} vox")

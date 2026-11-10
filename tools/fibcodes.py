"""fibcodes.py AFV CODES.raw --box ... [--tr T] [--rad 2]: same-sheet test on the labelled surfaces only (no band phase): fibre samples
within --rad voxels of a GT winding-code voxel (code = 1 + round(18 q) mod 252, q in turns mod 14) take that q; same-fibre pairs at arc
distance d agree when |dq| < 0.5 turn. Also the share of fibre samples near a code voxel, per family. (analysis; python3 -I)"""
import argparse, sys
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
import numpy as np
from fiblib import Map, open_afv, runs_in_box
ap = argparse.ArgumentParser(); ap.add_argument("afv"); ap.add_argument("codes"); ap.add_argument("--box", required=True); ap.add_argument("--level", type=int, default=2)
ap.add_argument("--tr"); ap.add_argument("--rad", type=int, default=2); ap.add_argument("--tag", default="box"); a = ap.parse_args()
box = tuple(map(int, a.box.split(","))); bz, by, bx, nz, ny, nx = box; R = a.rad
cd = np.fromfile(a.codes, dtype=np.uint8).reshape(nz, ny, nx)
mp = Map(a.level, a.tr); db, vs = open_afv(a.afv); runs = runs_in_box(db, mp, box)
offs = sorted([(dz, dy, dx) for dz in range(-R, R + 1) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if dz * dz + dy * dy + dx * dx <= R * R], key=lambda o: sum(k * k for k in o))
D = (16, 32, 64, 128, 256, 512); P = np.zeros((3, len(D), 3)); near = np.zeros((3, 2))
for fid, fam, flen, r, tg in runs:
    q = np.rint(r - [bz, by, bx]).astype(np.int64); inb = np.all((q >= R) & (q < np.array([nz, ny, nx]) - R), axis=1)
    c = np.zeros(len(r), np.int32); idx = np.flatnonzero(inb)
    for o in offs:
        todo = idx[c[idx] == 0]
        if len(todo) == 0: break
        v = cd[tuple((q[todo] + o).T)]; c[todo] = np.where((v >= 1) & (v <= 252), v, 0)
    near[fam] += [inb.sum(), (c > 0).sum()]
    w = np.where(c > 0, (c - 1) / 18.0, np.nan)
    for di, d in enumerate(D):
        if len(w) <= d: continue
        dq = w[d::4] - w[:-d:4]; dq = dq[np.isfinite(dq)]; dq = np.abs((dq + 7) % 14 - 7)
        P[fam, di] += [len(dq), np.sum(dq < 0.25), np.sum(dq < 0.5)]
for f, fn in enumerate("HV?"):
    if near[f, 0] == 0: continue
    print(f"{a.tag} {fn}: samples {int(near[f, 0])}, within {R} voxels of a code voxel {near[f, 1] / near[f, 0]:.3f}; same-sheet pairs (|dq| < 0.5): " +
          "  ".join(f"d{d}: {int(P[f, di, 2])}/{int(P[f, di, 0])} = {P[f, di, 2] / max(1, P[f, di, 0]):.4f}" for di, d in enumerate(D)))

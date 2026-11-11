"""fibwrap.py AFV WRAPS.f32 --slab z0,nz,tile --region y0,y1,x0,x1 [--level 0] [--tr T] --um U --part i/n --out rows.tsv
The fibre judge against the final wraps of a slab (analysis; python3 -I): WRAPS.f32 from `ufsm wrap-dump` (z, y, x, id, column / width
in the CT's level voxels). Per tile the band phase of each fibre sample between its two nearest wrap sheets (fiblib.wrap_phase) is
unwrapped along the fibre exactly as for a surf-labels phase raster (fiblib.phase_stats). Rows in fibjudge.py's "our" columns."""
import argparse, os, sys
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
import numpy as np
from scipy.spatial import cKDTree
from fiblib import Map, open_afv, runs_in_box, phase_stats, wrap_phase
ap = argparse.ArgumentParser(); ap.add_argument("afv"); ap.add_argument("wraps"); ap.add_argument("--slab", required=True); ap.add_argument("--region", required=True)
ap.add_argument("--level", type=int, default=0); ap.add_argument("--tr"); ap.add_argument("--um", type=float, required=True); ap.add_argument("--part", default="0/1"); ap.add_argument("--out", required=True)
ap.add_argument("--first", type=int, default=-10**9); ap.add_argument("--last", type=int, default=10**9)   # wrap ids kept (the labels' trim)
a = ap.parse_args(); D = (16, 32, 64, 128, 256); M = 44
z0, nz, t = map(int, a.slab.split(",")); y0, y1, x0, x1 = map(int, a.region.split(","))
boxes = [(z0, y, x, nz, t, t) for y in range(y0, y1 - t + 1, t) for x in range(x0, x1 - t + 1, t)]; pi, pn = map(int, a.part.split("/")); boxes = boxes[pi::pn]
V = np.fromfile(a.wraps, dtype=np.float32).reshape(-1, 5); V = V[(V[:, 3] >= a.first) & (V[:, 3] <= a.last) & (V[:, 4] > 0)]   # (column 0 is a closing duplicate)
mp = Map(a.level, a.tr); db, vs = open_afv(a.afv)
hdr = ["box", "runs", "pts"] + [f"our_{fn}_{c}" for fn in "HVU" for c in ["arc_m", "sw", "net"] + [f"p{d}" for d in D]] + ["our_known", "nvert"]
new = not os.path.exists(a.out)
with open(a.out, "a") as o:
    if new: o.write("\t".join(hdr) + "\n")
    for box in boxes:
        m = np.all((V[:, :3] >= np.array(box[:3]) - M) & (V[:, :3] < np.array(box[:3]) + np.array(box[3:]) + M), axis=1)
        if m.sum() < 5000: continue
        v = V[m]; par = (v[:, 3].astype(np.int64) & 1) == 1
        if par.sum() < 1000 or (~par).sum() < 1000: continue
        trees = tuple((cKDTree(v[k, :3].astype(np.float64)), (-v[k, 3] + v[k, 4]).astype(np.float64)) for k in (~par, par))
        runs = runs_in_box(db, mp, box)
        s = phase_stats(runs, None, box, D, sampler=lambda p: wrap_phase(trees, p))
        row = [",".join(map(str, box)), str(len(runs)), str(s["pts"])]
        for f in (0, 1, 2):
            row += [f"{s['arc'][f] * a.um * 1e-6:.4f}", f"{s['sw'][f]:.0f}", f"{s['net'][f]:.0f}"] + [f"{s['pair'][f, di, 0]:.0f}:{s['pair'][f, di, 2]:.0f}" for di in range(len(D))]
        row += [f"{s['lab'] / max(1, s['pts']):.3f}", str(int(m.sum()))]
        o.write("\t".join(row) + "\n"); o.flush(); print(row[0], "runs", row[1], "known", row[-2], "H arc", row[3], "sw", row[4], flush=True)

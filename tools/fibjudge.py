"""fibjudge.py AFV --level L [--tr T] --ours ROOT --gt ROOT GTPHASE GTCODES --boxes "z,y,x,nz,ny,nx;..." | --slab z0,nz,tile --part i/n --out rows.tsv
Fibres as a judge of a winding labelling (analysis only; python3 -I): per box, the fibre sheet-switch statistics against OUR phase
raster (ROOT/phase.zarr/KEY from surf-labels) and, where given, against the GT phase, plus the GT-based offset-cluster score of our
labels (fiblib.offset_clusters). One TSV row per box."""
import argparse, os, subprocess, sys, tempfile
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
import numpy as np
from fiblib import Map, open_afv, runs_in_box, phase_stats, offset_clusters

ap = argparse.ArgumentParser()
ap.add_argument("afv"); ap.add_argument("--level", type=int, default=2); ap.add_argument("--tr"); ap.add_argument("--um", type=float, default=9.6)
ap.add_argument("--ours", nargs=3, metavar=("ROOT", "PHASEKEY", "CODESKEY")); ap.add_argument("--gt", nargs=4, metavar=("PROOT", "PKEY", "CROOT", "CKEY"))
ap.add_argument("--boxes"); ap.add_argument("--slab"); ap.add_argument("--region", help="y0,y1,x0,x1 limits of the --slab tiles"); ap.add_argument("--shape", default="18946,8174,8174"); ap.add_argument("--part", default="0/1")
ap.add_argument("--ours-raw", nargs=3, metavar=("RAW", "BOX", "STEP"), help="our phase from fit-phase-box: file, its box z,y,x,nz,ny,nx, its step")
ap.add_argument("--out", required=True); ap.add_argument("--ufsm", default="/home/forrest/ufsm-lat/build/ufsm"); ap.add_argument("--tmp", required=True)
a = ap.parse_args()
D = (16, 32, 64, 128, 256)
OR = None
if a.ours_raw:
    ob = tuple(map(int, a.ours_raw[1].split(","))); OS = int(a.ours_raw[2]); osh = tuple((n + OS - 1) // OS for n in ob[3:]); OR = np.memmap(a.ours_raw[0], dtype=np.uint8, mode="r", shape=osh)
if a.boxes: boxes = [tuple(map(int, b.split(","))) for b in a.boxes.split(";")]
else:
    z0, nz, t = map(int, a.slab.split(",")); Z, Y, X = map(int, a.shape.split(","))
    y0, y1, x0, x1 = map(int, a.region.split(",")) if a.region else (0, Y, 0, X)
    boxes = [(z0, y, x, nz, t, t) for y in range(y0, y1 - t + 1, t) for x in range(x0, x1 - t + 1, t)]
pi, pn = map(int, a.part.split("/")); boxes = boxes[pi::pn]
mp = Map(a.level, a.tr); db, vs = open_afv(a.afv)
def read(root, key, box, name):
    f = os.path.join(a.tmp, f"{name}.{os.getpid()}.raw")
    r = subprocess.run([a.ufsm, "read", root, key, *map(str, box), f, "--threads", "1"], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-300:])
    v = np.fromfile(f, dtype=np.uint8).reshape(box[3:]); os.unlink(f); return v
def fmt(s):
    o = []
    for f in (0, 1, 2):
        A = s["arc"][f] * a.um * 1e-6
        o += [f"{A:.4f}", f"{s['sw'][f]:.0f}", f"{s['net'][f]:.0f}"] + [f"{s['pair'][f, di, 0]:.0f}:{s['pair'][f, di, 2]:.0f}" for di in range(len(D))]
    return o
hdr = ["box", "runs", "pts"] + [f"{w}_{fn}_{c}" for w in ("our", "gt") for fn in "HVU" for c in ["arc_m", "sw", "net"] + [f"p{d}" for d in D]] + ["oc_n", "oc_ids", "oc_pair", "oc_R", "oc_sgn", "our_known", "gt_known"]
new = not os.path.exists(a.out)
with open(a.out, "a") as o:
    if new: o.write("\t".join(hdr) + "\n")
    for box in boxes:
        lo, hi = mp.box_afv(box)
        nb = db.execute("select count(*) from block_bounds where max_x >= ? and min_x <= ? and max_y >= ? and min_y <= ? and max_z >= ? and min_z <= ?", (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])).fetchone()[0]
        if nb < 200: continue
        if OR is not None:
            lo3 = [(box[d] - ob[d]) // OS for d in range(3)]; hi3 = [(box[d] + box[3 + d] - ob[d] + OS - 1) // OS for d in range(3)]
            if min(lo3) < 0 or any(hi3[d] > osh[d] for d in range(3)): continue
            op = np.ascontiguousarray(OR[lo3[0]:hi3[0], lo3[1]:hi3[1], lo3[2]:hi3[2]])
        else: op = read(a.ours[0], a.ours[1], box, "op")
        ok_o = float((op <= 251).mean())
        gp = read(a.gt[0], a.gt[1], box, "gp") if a.gt else None; ok_g = float((gp <= 251).mean()) if a.gt else 0
        if ok_o < 0.02 and ok_g < 0.02: continue
        runs = runs_in_box(db, mp, box)
        so = phase_stats(runs, op, box, D, S=OS if OR is not None else 1); row = [",".join(map(str, box)), str(len(runs)), str(so["pts"])] + fmt(so)
        if a.gt:
            sg = phase_stats(runs, gp, box, D); row += fmt(sg)
            oc = None
            if ok_o >= 0.02 and ok_g >= 0.02:
                gc = read(a.gt[2], a.gt[3], box, "gc"); ocd = read(a.ours[0], a.ours[2], box, "oc"); oc = offset_clusters(gc, ocd, op)
            row += [f"{x:.4f}" if isinstance(x, float) else str(x) for x in (oc or ("na",) * 5)]
        else: row += ["na"] * (len(fmt(so)) + 5)
        row += [f"{ok_o:.3f}", f"{ok_g:.3f}"]
        o.write("\t".join(row) + "\n"); o.flush()
        print(row[0], "runs", row[1], "our H arc", row[3], "sw", row[4], "oc", row[-7:-2], flush=True)

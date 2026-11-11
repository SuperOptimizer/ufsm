"""Picture evidence for the fibre judge: CT crops at native resolution with our wraps, the traced fibres and the places where the judge
counts a sheet change. analysis only (python3 -I); reads the public native CT chunks (zarr v2, raw 128^3 uint8) with plain curl.

usage: fibview.py NAME CT_URL WRAPS_DIR FIRST LAST FILE.fib UM Z0,Y0,X0 OUT_DIR [NZ_VIEWS NV_VIEWS]
  Z0,Y0,X0: a 512^3 tile of the level-2 grid (the judge's tile). UM: native voxel size.
Per tile: the judge's own measure is repeated here point by point (wraps basis: nearest vertex per wrap-id parity, phase unwrapped along
the fibre, 9-sample median, hysteresis 0.75 turn), which gives the positions of the sheet changes. Views: z slices centred on sheet
changes of H fibres, and vertical slices (z up, the horizontal axis y or x, whichever is nearer the sheet normal) centred on sheet
changes of V fibres. 768 x 768 px = 768 native voxels."""
import sys, os, struct, subprocess
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fiblib import wrap_phase

name, url, wdir, first, last, fibp, um, tile, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], float(sys.argv[7]), sys.argv[8], sys.argv[9]
NZV = int(sys.argv[10]) if len(sys.argv) > 10 else 2; NVV = int(sys.argv[11]) if len(sys.argv) > 11 else 2
tz, ty, tx = map(int, tile.split(",")); T = 512; L = 4.0; PX = 768
U = "/home/forrest/ufsm-fib/build/ufsm"; CACHE = "/vesuvius/ufsm/fibers/raw/ctcache/" + url.rstrip("/").split("/")[-2]; TMP = out + "/tmp"; os.makedirs(TMP, exist_ok=True); os.makedirs(out, exist_ok=True)

def wraps(z0, z1, y0, y1, x0, x1, grid_um, tag):
    f = f"{TMP}/{name}_{tag}.f32"
    subprocess.run([U, "wrap-dump", wdir, f, "--z", f"{z0},{z1}", "--yx", f"{y0},{y1},{x0},{x1}", "--every", "1", "--grid-um", str(grid_um)], check=True, stdout=subprocess.DEVNULL)
    v = np.fromfile(f, np.float32).reshape(-1, 5); os.remove(f); return v[(v[:, 3] >= first) & (v[:, 3] <= last) & (v[:, 4] > 0)]

def ct_box(z0, y0, x0, n):   # native box, n = (nz, ny, nx)
    a = np.zeros(n, np.uint8); need = []
    for cz in range(z0 // 128, (z0 + n[0] - 1) // 128 + 1):
        for cy in range(y0 // 128, (y0 + n[1] - 1) // 128 + 1):
            for cx in range(x0 // 128, (x0 + n[2] - 1) // 128 + 1):
                p = f"{CACHE}/{cz}/{cy}/{cx}"
                if not os.path.exists(p): os.makedirs(os.path.dirname(p), exist_ok=True); need.append((p, f"{url}/{cz}/{cy}/{cx}"))
    for i in range(0, len(need), 8):
        ps = [subprocess.Popen(["curl", "-s", "-f", "--retry", "3", "-o", p + ".part", u]) for p, u in need[i:i + 8]]
        for (p, u), q in zip(need[i:i + 8], ps):
            if q.wait() == 0: os.replace(p + ".part", p)
            else: open(p, "wb").close()   # a missing chunk = fill 0
    for cz in range(z0 // 128, (z0 + n[0] - 1) // 128 + 1):
        for cy in range(y0 // 128, (y0 + n[1] - 1) // 128 + 1):
            for cx in range(x0 // 128, (x0 + n[2] - 1) // 128 + 1):
                p = f"{CACHE}/{cz}/{cy}/{cx}"
                if os.path.getsize(p) != 128 ** 3: continue
                c = np.fromfile(p, np.uint8).reshape(128, 128, 128); o = np.array([cz, cy, cx]) * 128 - [z0, y0, x0]
                lo = np.maximum(o, 0); hi = np.minimum(o + 128, n)
                if np.all(hi > lo): a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = c[lo[0] - o[0]:hi[0] - o[0], lo[1] - o[1]:hi[1] - o[1], lo[2] - o[2]:hi[2] - o[2]]
    return a

# ---- fibres of the tile (native z, y, x) ----
f = open(fibp, "rb"); assert f.read(4) == b"UFIB"; _, n, npts = struct.unpack("<iii", f.read(12)); off = np.frombuffer(f.read(4 * (n + 1)), "<i4"); fam = np.frombuffer(f.read(n), np.uint8); P = np.frombuffer(f.read(12 * npts), "<f4").reshape(-1, 3).astype(np.float64)
lo = (np.array([tz, ty, tx]) + 0.5) * L - 0.5; hi = (np.array([tz + T, ty + T, tx + T]) + 0.5) * L - 0.5
runs = []
for r in range(n):
    p = P[off[r]:off[r + 1]]
    if len(p) > 2 and np.all(p.max(0) >= lo - 400) and np.all(p.min(0) <= hi + 400): runs.append((chr(fam[r]), p))
# ---- the judge, point by point (level-2 voxels) ----
M = 44; V2 = wraps(tz - M, tz + T + M, ty - M, ty + T + M, tx - M, tx + T + M, um * L, "l2"); par = (V2[:, 3].astype(int) & 1) == 1
trees = tuple((cKDTree(V2[k, :3].astype(np.float64)), (-V2[k, 3] + V2[k, 4]).astype(np.float64)) for k in (~par, par))
events = []   # (family, native zyx, k, stretch net, run index)
arc = {"H": 0.0, "V": 0.0}
for ri, (fm, p) in enumerate(runs):
    q = (p + 0.5) / L - 0.5; s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    if s[-1] < 2: continue
    t = np.arange(0, s[-1], 1.0); r = np.stack([np.interp(t, s, q[:, k]) for k in range(3)], 1)
    inb = np.all((np.rint(r) >= [tz, ty, tx]) & (np.rint(r) < [tz + T, ty + T, tx + T]), axis=1)
    ph = np.full(len(r), 255, np.uint8); ph[inb] = wrap_phase(trees, r[inb]); ok = ph <= 251; i = 0
    while i < len(r):
        if not ok[i]: i += 1; continue
        j = i
        while j < len(r) and ok[j]: j += 1
        if j - i >= 16:
            f_ = ph[i:j] / 252.0; d = np.diff(f_); u = np.concatenate([[0], np.cumsum(d - np.floor(d + 0.5))]); u = median_filter(u, 9, mode="nearest"); b = u[0]; ev = []
            for k in range(len(u)):
                if abs(u[k] - b) > 0.75: kk = round(u[k] - b); ev.append((i + k, kk)); b += kk
            net = sum(e[1] for e in ev); arc[fm if fm in arc else "H"] += (j - i)
            for (idx, kk) in ev: events.append((fm, (r[idx] + 0.5) * L - 0.5, kk, net, ri))
        i = j
npers = sum(1 for e in events if e[3] != 0)
print(f"{name} tile {tile}: {len(runs)} streamlines near the tile, labelled arc H {arc['H'] * um * L * 1e-6:.2f} m V {arc['V'] * um * L * 1e-6:.2f} m, {len(events)} sheet changes ({npers} in stretches with a net change)")

PAL = [(255, 80, 80), (80, 220, 80), (90, 140, 255), (255, 160, 40), (220, 90, 255), (60, 220, 220)]
def render(kind, c, ev, tag):
    """kind 'z': the slice z = c[0], axes (y down, x right). 'y' / 'x': vertical slice at fixed x / y, axes (z down, y or x right)."""
    c = np.rint(c).astype(int); h = PX // 2
    if kind == "z": o = np.array([c[0], c[1] - h, c[2] - h]); nn = (1, PX, PX); ax = (1, 2); fix = 0
    elif kind == "y": o = np.array([c[0] - h, c[1] - h, c[2]]); nn = (PX, PX, 1); ax = (0, 1); fix = 2   # normal along y: plane (z, y) at fixed x
    else: o = np.array([c[0] - h, c[1], c[2] - h]); nn = (PX, 1, PX); ax = (0, 2); fix = 1
    o = np.maximum(o, 0); img = ct_box(int(o[0]), int(o[1]), int(o[2]), nn).reshape(PX, PX)
    im = Image.fromarray(img).convert("RGB"); dr = ImageDraw.Draw(im)
    # wraps: vertices within 24 native voxels of the plane, thinned below to one row (z slice) or one vertex per row (vertical slice)
    tol = 24.0; b0 = o.astype(float).copy(); b1 = o + np.array(nn, float); b0[fix] = c[fix] - tol; b1[fix] = c[fix] + tol + 1
    W = wraps(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2], um, "nat")
    for wid in np.unique(W[:, 3]):
        w = W[W[:, 3] == wid]; col = PAL[int(wid) % len(PAL)]; zr = np.rint(w[:, 0] / 2).astype(int)   # rows are constant z
        if kind == "z":   # the row nearest the slice
            rows = np.unique(zr); w = w[zr == rows[np.argmin(np.abs(rows * 2 - c[0]))]]
        else:             # per row the vertex nearest the plane
            od = np.lexsort((np.abs(w[:, fix] - c[fix]), zr)); w = w[od]; zr = zr[od]; w = w[np.concatenate([[True], zr[1:] != zr[:-1]])]
        xy = np.stack([w[:, ax[1]] - o[ax[1]], w[:, ax[0]] - o[ax[0]]], 1)
        # vertices in spiral order (z slice: along the row) or in z order (vertical slice); neighbours joined when close
        od = np.argsort(w[:, 4] if kind == "z" else w[:, 0], kind="stable"); xy = xy[od]
        for i in range(len(xy) - 1):
            if np.hypot(*(xy[i + 1] - xy[i])) < 48: dr.line([tuple(map(float, xy[i])), tuple(map(float, xy[i + 1]))], fill=col, width=1)
        k = np.argmin(np.abs(xy[:, 0] - PX / 2) + np.abs(xy[:, 1] - PX / 2) * 0.2) if len(xy) else None
        if k is not None and 8 < xy[k, 0] < PX - 30 and 8 < xy[k, 1] < PX - 12: dr.text((float(xy[k, 0]) + 3, float(xy[k, 1]) + 2), str(int(wid)), fill=col)
    # fibres: the pieces within 20 native voxels of the plane; the event's own fibre in white over them
    for ri, (fm, p) in enumerate(runs):
        near = np.abs(p[:, fix] - c[fix]) <= 20; col = (255, 255, 0) if fm == "H" else (0, 255, 255)
        if ri == ev[4]: col = (255, 255, 255)
        for i in range(len(p) - 1):
            if not (near[i] and near[i + 1]) and ri != ev[4]: continue
            a = (float(p[i, ax[1]] - o[ax[1]]), float(p[i, ax[0]] - o[ax[0]])); b = (float(p[i + 1, ax[1]] - o[ax[1]]), float(p[i + 1, ax[0]] - o[ax[0]]))
            if ri == ev[4] and not (near[i] and near[i + 1]): dr.line([a, b], fill=(150, 150, 150), width=1)   # the fibre away from the plane: grey
            else: dr.line([a, b], fill=col, width=2 if ri == ev[4] else 1)
    for e in events:
        if abs(e[1][fix] - c[fix]) > 20: continue
        x, y = float(e[1][ax[1]] - o[ax[1]]), float(e[1][ax[0]] - o[ax[0]]); r_ = 9 if e is ev else 6
        dr.ellipse([x - r_, y - r_, x + r_, y + r_], outline=(255, 0, 0) if e[3] != 0 else (255, 255, 255), width=2)
    dr.line([(20, PX - 14), (20 + 100.0 / um, PX - 14)], fill=(255, 255, 255), width=2); dr.text((20, PX - 30), "100 um", fill=(255, 255, 255))
    fn = f"{name}_{tile.replace(',', '-')}_{tag}.png"; im.save(f"{out}/{fn}")
    what = {"z": f"z slice, native z {c[0]} (y down, x right)", "y": f"vertical slice at native x {c[2]} (z down, y right)", "x": f"vertical slice at native y {c[1]} (z down, x right)"}[kind]
    with open(out + "/index.txt", "a") as ix:
        ix.write(f"{fn}\t{name} tile z,y,x {tile} (level 2, 512^3) centre native {c[0]},{c[1]},{c[2]}: {what}; {PX} px = {PX * um / 1000:.2f} mm; grey = CT; thin coloured lines + numbers = our wraps "
                 f"(colour by wrap index mod 6); yellow = traced H fibres, cyan = traced V fibres within 20 voxels ({20 * um:.0f} um) of the slice; white = the fibre this view is centred on (grey where it leaves the slice); "
                 f"red circle = the judge counts a sheet change there on a stretch whose net change is not zero (persistent), white circle = on a stretch that returns; big circle = this view's event "
                 f"({ev[0]} fibre, change {ev[2]:+d} turn, stretch net {ev[3]:+d})\n")
    print("  ", fn)

def pick(fm, k):
    c = [e for e in events if e[0] == fm and e[3] != 0 and np.all(e[1] > lo + PX / 2 + 8) and np.all(e[1] < hi - PX / 2 - 8)]
    if not c: c = [e for e in events if e[0] == fm and e[3] != 0]
    sel = []; rng = np.random.default_rng(7); order = rng.permutation(len(c))
    for i in order:
        if all(np.linalg.norm(c[i][1] - s[1]) > 500 for s in sel): sel.append(c[i])
        if len(sel) == k: break
    return sel
for i, e in enumerate(pick("H", NZV)): render("z", e[1], e, f"zslice{i + 1}")
for i, e in enumerate(pick("V", NVV)):
    # the sheet normal's horizontal direction at the event: from the nearest vertex of one parity to the other's
    # the sheet normal at the event: the smallest principal axis of the nearest wrap's vertices within 30 level-2 voxels
    q = (e[1] + 0.5) / L - 0.5; (ta, _), (tb, _) = trees; da, _ = ta.query(q); db, _ = tb.query(q); tr_ = ta if da <= db else tb
    nb = tr_.data[tr_.query_ball_point(q, 30.0)]; d = np.linalg.svd(nb - nb.mean(0), full_matrices=False)[2][2] if len(nb) >= 8 else np.array([0, 1.0, 0])
    render("y" if abs(d[1]) >= abs(d[2]) else "x", e[1], e, f"vslice{i + 1}")

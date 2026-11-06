# cutpatch.py BOXDIR Z0 Y0 X0 N AXIS AXS SGN: impurity diagnosis + GT-free cut test on VC3D patches (BOXDIR/paths).
# per grid vertex: GT sheet (nearest code within 1), ridge support (ridge within R voxels), m7 prob.
# cuts: keep vertices with support, split into 4-connected components; report purity / kept area vs uncut.
import sys, os, json, glob, numpy as np, tifffile
from scipy import ndimage
bd = sys.argv[1]; z0, y0, x0, n = map(int, sys.argv[2:6]); axp, axs, sgn = sys.argv[6], float(sys.argv[7]), float(sys.argv[8])
gt = np.fromfile(bd + "/gt.raw", np.uint8).reshape(n, n, n); gs = (gt >= 1) & (gt <= 252)
gtn = ndimage.maximum_filter(np.where(gs, gt, 0), size=3)
rid = np.fromfile(bd + "/ridge.raw", np.uint8).reshape(n, n, n) > 0
prob = np.fromfile(bd + "/prob.raw", np.uint8).reshape(n, n, n)
sup = {r: ndimage.binary_dilation(rid, iterations=r) for r in (1, 2)}
cp = json.load(open(axp))["control_points"]; cz = np.array([c["z"] for c in cp]) * axs; cy = np.array([c["y"] for c in cp]) * axs; cx = np.array([c["x"] for c in cp]) * axs
o = np.argsort(cz); cz, cy, cx = cz[o], cy[o], cx[o]
def sheet(z, y, x, c):
    th = np.arctan2(y + y0 - np.interp(z + z0, cz, cy), x + x0 - np.interp(z + z0, cz, cx))
    tr = np.arctan2(np.sin(th).mean(), np.cos(th).mean()); th = tr + np.angle(np.exp(1j * (th - tr)))
    return np.mod(np.rint((c - 1) / 18.0 - sgn * th / (2 * np.pi)), 14).astype(int)
def pur(s):
    if len(s) == 0: return 0, 0
    h = np.bincount(s, minlength=14); return h.max(), h.sum()
tot = {}; stats = {"sup_bad": 0, "sup_good": 0, "uns_bad": 0, "uns_good": 0}; pb = {"bad": [], "good": []}
for p in sorted(glob.glob(bd + "/paths/*")):
    if not os.path.exists(p + "/x.tif"): continue
    X, Y, Z = (tifffile.imread(p + "/" + c + ".tif").astype(np.float32) for c in "xyz")
    v = X > -1; xi, yi, zi = (np.rint(A).astype(np.int64) for A in (X, Y, Z))
    v &= (xi >= 0) & (yi >= 0) & (zi >= 0) & (xi < n) & (yi < n) & (zi < n)
    xi, yi, zi = np.where(v, xi, 0), np.where(v, yi, 0), np.where(v, zi, 0)
    c = gtn[zi, yi, xi]; m = v & (c > 0)
    if m.sum() < 4: continue
    S = np.full(v.shape, -1); S[m] = sheet(zi[m].astype(float), yi[m].astype(float), xi[m].astype(float), c[m].astype(float))
    dom = np.bincount(S[m], minlength=14).argmax(); bad = m & (S != dom)
    for r in (1, 2):
        sp = sup[r][zi, yi, xi] & v
        if r == 1:
            stats["sup_bad"] += (bad & sp).sum(); stats["uns_bad"] += (bad & ~sp).sum()
            stats["sup_good"] += (m & ~bad & sp).sum(); stats["uns_good"] += (m & ~bad & ~sp).sum()
            pb["bad"] += list(prob[zi, yi, xi][bad]); pb["good"] += list(prob[zi, yi, xi][m & ~bad])
        for mode in ("uncut", "sup", "sup_er1"):
            if mode == "uncut": comps = [(m, 1)]
            else:
                k = sp if mode == "sup" else ndimage.binary_erosion(sp, iterations=1)
                lab, nl = ndimage.label(k); comps = [(m & (lab == i), 1) for i in range(1, nl + 1)]
            for cm, _ in comps:
                a, b = pur(S[cm]); key = (r, mode); t = tot.setdefault(key, [0, 0, 0]); t[0] += a; t[1] += b; t[2] += int(b > 0 and a / b < .95) * b
print("matched vertices: wrong-sheet %d (%.1f%% of them unsupported by ridge<=1), right-sheet %d (%.1f%% unsupported)" % (
    stats["sup_bad"] + stats["uns_bad"], 100 * stats["uns_bad"] / max(1, stats["sup_bad"] + stats["uns_bad"]),
    stats["sup_good"] + stats["uns_good"], 100 * stats["uns_good"] / max(1, stats["sup_good"] + stats["uns_good"])))
print("m7 prob median wrong %.0f right %.0f" % (np.median(pb["bad"]) if pb["bad"] else -1, np.median(pb["good"])))
base = tot[(1, "uncut")][1]
for (r, mode), (a, b, lowb) in sorted(tot.items()):
    if mode == "uncut" and r == 2: continue
    print("  ridge<=%d %-8s purity %.4f  kept %.3f  area in comps <.95: %.3f" % (r, mode, a / max(1, b), b / max(1, base), lowb / max(1, b)))

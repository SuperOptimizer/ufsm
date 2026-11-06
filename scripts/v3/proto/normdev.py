# normdev.py BOXDIR Z0 Y0 X0 N AXIS AXS SGN: at patch vertices near a GT sheet change (transition) vs elsewhere,
# |cos| between the patch normal and the data sheet normal (structure tensor of m7 prob, sigma 2 grad / 4 window); and cut test on it.
import sys, os, json, glob, numpy as np, tifffile
from scipy import ndimage
bd = sys.argv[1]; z0, y0, x0, n = map(int, sys.argv[2:6]); axp, axs, sgn = sys.argv[6], float(sys.argv[7]), float(sys.argv[8])
gt = np.fromfile(bd + "/gt.raw", np.uint8).reshape(n, n, n); gs = (gt >= 1) & (gt <= 252)
gtn = ndimage.maximum_filter(np.where(gs, gt, 0), size=3)
P = np.fromfile(bd + "/prob.raw", np.uint8).reshape(n, n, n).astype(np.float32)
g = [ndimage.gaussian_filter(P, 2, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
T = {}
for i in range(3):
    for j in range(i, 3): T[i, j] = ndimage.gaussian_filter(g[i] * g[j], 4)
del g
rid = ndimage.binary_dilation(np.fromfile(bd + "/ridge.raw", np.uint8).reshape(n, n, n) > 0, iterations=1)
cp = json.load(open(axp))["control_points"]; cz = np.array([c["z"] for c in cp]) * axs; cy = np.array([c["y"] for c in cp]) * axs; cx = np.array([c["x"] for c in cp]) * axs
o = np.argsort(cz); cz, cy, cx = cz[o], cy[o], cx[o]
def sheet(z, y, x, c):
    th = np.arctan2(y + y0 - np.interp(z + z0, cz, cy), x + x0 - np.interp(z + z0, cz, cx))
    tr = np.arctan2(np.sin(th).mean(), np.cos(th).mean()); th = tr + np.angle(np.exp(1j * (th - tr)))
    return np.mod(np.rint((c - 1) / 18.0 - sgn * th / (2 * np.pi)), 14).astype(int)
ct, cn = [], []; res = {}
for p in sorted(glob.glob(bd + "/paths/*")):
    if not os.path.exists(p + "/x.tif"): continue
    X, Y, Z = (tifffile.imread(p + "/" + c + ".tif").astype(np.float32) for c in "xyz")
    v = X > -1; xi, yi, zi = (np.rint(A).astype(np.int64) for A in (X, Y, Z))
    v &= (xi >= 0) & (yi >= 0) & (zi >= 0) & (xi < n) & (yi < n) & (zi < n)
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = False
    xi, yi, zi = np.where(v, xi, 0), np.where(v, yi, 0), np.where(v, zi, 0)
    # patch normal from central differences (zyx)
    Pz = np.stack([Z, Y, X], -1)
    du = np.roll(Pz, -1, 1) - np.roll(Pz, 1, 1); dv = np.roll(Pz, -1, 0) - np.roll(Pz, 1, 0)
    nv = np.cross(du, dv); nv /= np.linalg.norm(nv, axis=-1, keepdims=True) + 1e-9
    vv = v & np.roll(v, 1, 0) & np.roll(v, -1, 0) & np.roll(v, 1, 1) & np.roll(v, -1, 1)
    idx = (zi[vv], yi[vv], xi[vv])
    M = np.zeros((vv.sum(), 3, 3), np.float32)
    for (i, j), A in T.items(): M[:, i, j] = M[:, j, i] = A[idx]
    w, e = np.linalg.eigh(M); dn = e[:, :, 2]   # largest eigvec = sheet normal
    cs = np.full(v.shape, np.nan); cs[vv] = np.abs((dn * nv[vv]).sum(-1))
    c = gtn[zi, yi, xi]; m = vv & (c > 0)
    if m.sum() < 4: continue
    S = np.full(v.shape, -1); S[m] = sheet(zi[m].astype(float), yi[m].astype(float), xi[m].astype(float), c[m].astype(float))
    ch = np.zeros_like(m)
    for ax in (0, 1):
        a = np.roll(S, 1, ax); ch |= m & (a >= 0) & (a != S)
    tr_ = ndimage.binary_dilation(ch, iterations=2) & m
    ct += list(cs[tr_]); cn += list(cs[m & ~tr_])
    sp = rid[zi, yi, xi] & v
    for th in (0, 0.8, 0.9, 0.95):
        k = sp & (np.nan_to_num(cs, nan=0) >= th)
        lab, nl = ndimage.label(k)
        for i in range(1, nl + 1):
            s = S[m & (lab == i)]
            if len(s): h = np.bincount(s, minlength=14); r = res.setdefault(th, [0, 0]); r[0] += h.max(); r[1] += h.sum()
    r = res.setdefault("base", [0, 0]); h = np.bincount(S[m], minlength=14); r[0] += h.max(); r[1] += h.sum()
ct, cn = np.array(ct), np.array(cn)
print("|cos| patch vs data normal: transition median %.3f (<.9: %.2f)  elsewhere median %.3f (<.9: %.2f)" % (np.median(ct), (ct < .9).mean(), np.median(cn), (cn < .9).mean()))
b = res["base"][1]
for th in (0, 0.8, 0.9, 0.95): a, c_ = res[th]; print("  ridge-support + |cos|>=%.2f cut: purity %.4f kept %.3f" % (th, a / c_, c_ / b))

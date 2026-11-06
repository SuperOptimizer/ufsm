# psolve.py BOX Z0 Y0 X0 N AXIS AXS SGN [MINV 100]: prototype patch turn solver on a box.
# patches = VC3D paths cut at ridge-unsupported vertices (4-conn comps >= MINV vertices).
# rays: bins (z 4 vox, arc 4 vox); consecutive crossings of different comps along r -> edge k_b - k_a = 1 + (# ridge runs strictly between them).
# solve weighted L1 integer offsets (LS init, integer weighted-median sweeps). Score vs GT sheet ids (mod 14).
import sys, os, json, glob, numpy as np, tifffile
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse.csgraph import connected_components
bd = sys.argv[1]; z0, y0, x0, n = map(int, sys.argv[2:6]); axp, axs, sgn = sys.argv[6], float(sys.argv[7]), float(sys.argv[8]); MINV = int(sys.argv[9]) if len(sys.argv) > 9 else 100
gt = np.fromfile(bd + "/gt.raw", np.uint8).reshape(n, n, n); gs = (gt >= 1) & (gt <= 252); gtn = ndimage.maximum_filter(np.where(gs, gt, 0), size=3)
rr = np.fromfile(bd + "/ridge.raw", np.uint8).reshape(n, n, n) > 0; sup = ndimage.binary_dilation(rr, iterations=1)
cp = json.load(open(axp))["control_points"]; cz = np.array([c["z"] for c in cp]) * axs; cy = np.array([c["y"] for c in cp]) * axs; cx = np.array([c["x"] for c in cp]) * axs
o = np.argsort(cz); cz, cy, cx = cz[o], cy[o], cx[o]
def axis(z): return np.interp(z + z0, cz, cy) - y0, np.interp(z + z0, cz, cx) - x0
ay, ax = axis(np.array([n / 2])); t0 = np.arctan2(n / 2 - ay[0], n / 2 - ax[0])
def polar(z, y, x):
    a, b = axis(z); th = np.arctan2(y - a, x - b); return np.hypot(y - a, x - b), t0 + np.angle(np.exp(1j * (th - t0)))
V = []   # per comp: (z, y, x, r, th) arrays
for p in sorted(glob.glob(bd + "/paths/*")):
    if not os.path.exists(p + "/x.tif"): continue
    X, Y, Z = (tifffile.imread(p + "/" + c + ".tif").astype(np.float32) for c in "xyz")
    v = X > -1; xi, yi, zi = (np.rint(A).astype(np.int64) for A in (X, Y, Z))
    v &= (xi >= 0) & (yi >= 0) & (zi >= 0) & (xi < n) & (yi < n) & (zi < n)
    k = v & sup[np.where(v, zi, 0), np.where(v, yi, 0), np.where(v, xi, 0)]
    lab, nl = ndimage.label(k)
    for i in range(1, nl + 1):
        m = lab == i
        if m.sum() >= MINV: V.append((Z[m], Y[m], X[m]))
N = len(V); print("comps", N, "vertices", sum(len(a[0]) for a in V))
# GT sheet per comp
gtd = []
for z, y, x in V:
    zi, yi, xi = (np.rint(a).astype(int) for a in (z, y, x)); c = gtn[zi, yi, xi]; m = c > 0
    if m.sum() < 10: gtd.append((-1, 0, 0)); continue
    r, th = polar(z[m], y[m], x[m]); s = np.mod(np.rint((c[m] - 1) / 18.0 - sgn * th / (2 * np.pi)), 14).astype(int)
    h = np.bincount(s, minlength=14); gtd.append((h.argmax(), h.max(), h.sum()))
# ray crossings
rm = np.median(np.concatenate([polar(*a)[0] for a in V])); dth = 4.0 / rm
Z_, R_, T_, C_ = [], [], [], []
for i, (z, y, x) in enumerate(V):
    r, th = polar(z, y, x); Z_.append(z); R_.append(r); T_.append(th); C_.append(np.full(len(z), i))
Z_, R_, T_, C_ = map(np.concatenate, (Z_, R_, T_, C_))
zb = np.floor(Z_ / 4).astype(int); tb = np.floor(T_ / dth).astype(int); key = zb * 100000 + (tb - tb.min())
o = np.lexsort((R_, key)); key, R_, C_, Z_, T_ = key[o], R_[o], C_[o], Z_[o], T_[o]

P = np.fromfile(bd + "/prob.raw", np.uint8).reshape(n, n, n).astype(np.float32)
CT = ndimage.gaussian_filter(np.fromfile(bd + "/ctbox.raw", np.uint8).reshape(n, n, n).astype(np.float32), 1.0)
def runs(h): return int((np.diff(np.r_[0, h.astype(int)]) == 1).sum())
def peaks(v, prom):
    if len(v) < 3: return 0
    from scipy.signal import find_peaks
    return len(find_peaks(v, prominence=prom)[0])
CNT = ["ridge64", "probpk16", "probpk32", "ctpk4", "ctpk8", "ctpk12", "spacing"]
votes = {c: {} for c in CNT}; gapl = []
brk = np.flatnonzero(np.diff(key)) + 1; starts = np.r_[0, brk]; ends = np.r_[brk, len(key)]
for s, e in zip(starts, ends):
    if e - s < 2: continue
    # collapse runs of the same comp (take mean r)
    cc, rr_ = C_[s:e], R_[s:e]; zc = Z_[s:e].mean(); tc = T_[s:e].mean()
    seq = []
    for c, r in zip(cc, rr_):
        if seq and seq[-1][0] == c: continue
        if seq and r - seq[-1][1] < 3: continue
        seq.append((c, r))
    ay_, ax_ = axis(np.array([zc])); ay_, ax_ = ay_[0], ax_[0]
    for (a, ra), (b, rb) in zip(seq, seq[1:]):
        t = np.arange(ra + 2.5, rb - 2.5, 0.5)
        if len(t):
            yy = np.clip(np.rint(ay_ + t * np.sin(tc)).astype(int), 0, n - 1); xx = np.clip(np.rint(ax_ + t * np.cos(tc)).astype(int), 0, n - 1)
            hit = rr[int(round(zc)), yy, xx].astype(int); runs = int(((np.diff(np.r_[0, hit])) == 1).sum())
        else: runs = 0
        zz = int(round(zc)); res_ = {"ridge64": runs}
        if len(t):
            # interpolate along ray incl. the two crossings for peak finders (exclude peaks within 2.5 of the ends)
            tt = np.arange(ra, rb + 0.01, 0.5); yy2 = np.clip(np.rint(ay_ + tt * np.sin(tc)).astype(int), 0, n - 1); xx2 = np.clip(np.rint(ax_ + tt * np.cos(tc)).astype(int), 0, n - 1)
            pv = P[zz, yy2, xx2]; cv = CT[zz, yy2, xx2]; inner = (tt > ra + 2.5) & (tt < rb - 2.5)
            from scipy.signal import find_peaks
            def cntp(v, prom): pk = find_peaks(v, prominence=prom)[0]; return int(inner[pk].sum())
            res_.update(probpk16=cntp(pv, 16), probpk32=cntp(pv, 32), ctpk4=cntp(cv, 4), ctpk8=cntp(cv, 8), ctpk12=cntp(cv, 12))
        else: res_.update(probpk16=0, probpk32=0, ctpk4=0, ctpk8=0, ctpk12=0)
        gapl.append(rb - ra); res_["spacing"] = None
        for c_, val in res_.items():
            if val is None: continue
            votes[c_].setdefault((a, b), {}).setdefault(1 + val, 0); votes[c_][(a, b)][1 + val] += 1
        votes["spacing"].setdefault((a, b), []).append(rb - ra)

sp = np.median([g for g in gapl if g < np.percentile(gapl, 50)])  # typical single-sheet spacing ~ median of the lower half
print("spacing est %.1f vox" % sp)
okc = lambda i: gtd[i][0] >= 0 and gtd[i][1] / gtd[i][2] > .95
for c in CNT:
    tw = tw_ok = 0; hist = {}
    for (a, b), dd in votes[c].items():
        if c == "spacing": nvote = len(dd); val = max(1, int(round(np.median(dd) / sp)))
        else: nvote = sum(dd.values()); val = max(dd.items(), key=lambda kv: kv[1])[0]
        if nvote < 3 or not (okc(a) and okc(b)): continue
        tru = ((gtd[b][0] - gtd[a][0] + 7) % 14) - 7; e = val - tru
        hist[e] = hist.get(e, 0) + nvote; tw += nvote
    print("  %-9s correct %.3f  under %.3f  over %.3f" % (c, hist.get(0, 0) / tw, sum(v for k, v in hist.items() if k < 0) / tw, sum(v for k, v in hist.items() if k > 0) / tw))

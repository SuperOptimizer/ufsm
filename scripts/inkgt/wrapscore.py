#!/usr/bin/env python3
"""scripts/inkgt/wrapscore.py STRIPES-DIR SEGS.JSON OUT.TSV [--map DIR] [--um-wrap U] [--step K] [--tol 20,40,80] [--holdouts H.json]
Score an unroll's rendered ink against the published human ink labels mapped onto OUR flattened surfaces (2026-11-12, inkB).

Each published segment (SEGS.JSON: the ink-onfly segment list: tifxyz, labels, mask, validation) is sampled every K label px inside
its supervision mask. Each sample is a 3-D point on the published mesh (level-0 voxels: label px = canvas px, grid = canvas x scale,
bilinear in the tifxyz grid), with its label (ink / no ink), whether its validation raster holds it out, and whether it lies in an ink
holdout box (holdouts-ink.json, this scan's "ct"). Our stripes (STRIPES-DIR/<s>/x,y,z.tif in level-L voxels, meta voxel_size_um) are
scaled to level 0 (x um_wrap / um_label). Each sample finds the nearest grid vertex of any stripe (KD-tree), then the foot point on
that cell's bilinear patch (Gauss-Newton, 6 steps). Its distance d is the surface-to-surface miss. With d <= tol the sample takes the
value of our ink map at the foot point's canvas pixel (canvas = grid / scale; the map: --map DIR/<s>.tif or <s>_ink.jpg / <s>.jpg,
grey, any size proportional to the canvas) and the CT map's (<s>_ct.tif / <s>_ct.jpg, the floor).
Out per tolerance and split (all / validation / holdout / rest): labelled samples, coverage (share within tol), ink share, AUC and
best F1 of the ink map, the same for the CT, the median miss. The maps can be stretched: AUC and best F1 are rank-based (any
monotone stretch gives the same), F1 at a fixed threshold is not reported.
Memory: the stripes' vertices within the labelled bboxes (+ 200 voxels) only; ~50 bytes a vertex for the tree.
"""
import argparse, glob, json, os, sys
import numpy as np, tifffile, zarr
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("stripes"); ap.add_argument("segs"); ap.add_argument("out")
ap.add_argument("--map", default=None, help="dir with the ink renders (default: STRIPES-DIR/../../../img_fill)")
ap.add_argument("--um-label", type=float, default=None, help="the published meshes' voxel size (default: their meta, else 2.399)")
ap.add_argument("--step", type=int, default=8); ap.add_argument("--tol", default="20,40,80", help="miss tolerances in um")
ap.add_argument("--holdouts", default="/vesuvius/ufsm/inkgt/holdouts-ink.json"); ap.add_argument("--ct-key", default=None)
ap.add_argument("--vertices", default=None, help="coverage only from a vertex dump (ufsm wrap-dump: float32 z, y, x, id, col records) instead of stripes")
ap.add_argument("--vertex-um", type=float, default=9.596, help="the dump's voxel size")
ap.add_argument("--self-test", action="store_true", help="map onto the published meshes themselves (the labels as the map): AUC 1, miss 0")
a = ap.parse_args()
tols = [float(t) for t in a.tol.split(",")]

def rd2(p, step):   # a 2-D raster at 1/step (zarr array dir or tif) -> bool >= half
    if p.endswith((".tif", ".tiff")):
        x = tifffile.imread(p)
        if x.ndim == 3: x = x[..., 0] if x.shape[-1] in (3, 4) else x[0]
        x = x[::step, ::step]
    else:
        z = zarr.open_array(p, mode="r"); H, W = z.shape; rows = []
        for r in range(0, H, 4096): rows.append(np.asarray(z[r:min(H, r + 4096), :])[(-r) % step::step, ::step])
        x = np.concatenate(rows, 0)
    return x >= (1 if x.max() <= 1 else 128)

def load_xyz(d):
    m = json.load(open(os.path.join(d, "meta.json")))
    X, Y, Z = (tifffile.imread(os.path.join(d, c + ".tif")).astype(np.float32) for c in "xyz")
    return m, X, Y, Z

def bil(G, r, c):   # bilinear sample of grid G at float (r, c), arrays
    r0 = np.clip(np.floor(r).astype(np.int64), 0, G.shape[0] - 2); c0 = np.clip(np.floor(c).astype(np.int64), 0, G.shape[1] - 2)
    fr, fc = r - r0, c - c0
    return (1 - fr) * ((1 - fc) * G[r0, c0] + fc * G[r0, c0 + 1]) + fr * ((1 - fc) * G[r0 + 1, c0] + fc * G[r0 + 1, c0 + 1])

# ---- the labelled samples ----
J = json.load(open(a.segs)); hold = []
if a.holdouts and os.path.exists(a.holdouts):
    H = json.load(open(a.holdouts)); ck = a.ct_key
    if not ck: sys.exit("--ct-key KEY: the labels' CT (the holdout boxes of other scans share the coordinates)")
    hold = [b["l0"] for b in H["boxes"] if os.path.basename(b.get("ct", "")) == os.path.basename(ck)]
P, LAB, VAL, HOLD, SEG = [], [], [], [], []; seen = set(); segsurf = {}
for g in J["segments"]:
    key = (g["labels"], tuple(g.get("crop", [])))
    if key in seen: continue
    seen.add(key)
    m, X, Y, Z = load_xyz(g["tifxyz"]); sc = float(m["scale"][0]); um = a.um_label or float(m.get("voxel_size_um", 2.399))
    M = rd2(g["mask"], a.step); L = rd2(g["labels"], a.step); V = rd2(g["validation"], a.step) if g.get("validation") else np.zeros_like(M)
    h = min(M.shape[0], L.shape[0], V.shape[0]); w = min(M.shape[1], L.shape[1], V.shape[1]); M, L, V = M[:h, :w], L[:h, :w], V[:h, :w]
    if g.get("crop"):   # the label window [c0, r0, w, h] (label px): only its pixels (segments listed once per window)
        c0, r0, cw, chh = g["crop"]; win = np.zeros_like(M); win[r0 // a.step:(r0 + chh) // a.step, c0 // a.step:(c0 + cw) // a.step] = True; M = M & win
    rr, cc = np.nonzero(M)
    gr, gc = rr * a.step * sc, cc * a.step * sc   # label px -> canvas px (F = 1: label px = canvas px, ink-score's (c + 0.5) F - 0.5) -> grid (canvas x scale)
    ok = (gr >= 0) & (gc >= 0) & (gr <= Z.shape[0] - 1) & (gc <= Z.shape[1] - 1); gr, gc, rr, cc = gr[ok], gc[ok], rr[ok], cc[ok]
    r0 = np.floor(gr).astype(int).clip(0, Z.shape[0] - 2); c0 = np.floor(gc).astype(int).clip(0, Z.shape[1] - 2)
    valid = (Z[r0, c0] > 0) & (Z[r0 + 1, c0] > 0) & (Z[r0, c0 + 1] > 0) & (Z[r0 + 1, c0 + 1] > 0)
    gr, gc, rr, cc = gr[valid], gc[valid], rr[valid], cc[valid]
    p = np.stack([bil(Z, gr, gc), bil(Y, gr, gc), bil(X, gr, gc)], 1)
    P.append(p); LAB.append(L[rr, cc]); VAL.append(V[rr, cc]); SEG += [g["name"]] * len(rr)
    hb = np.zeros(len(rr), bool)
    for b in hold: hb |= np.all((p >= np.array(b[:3])) & (p < np.array(b[:3]) + np.array(b[3:])), 1)
    HOLD.append(hb)
    if a.self_test: segsurf[g["name"]] = (g, m, X, Y, Z, L, sc)
    print(f"labels: {g['name'][:44]:44s} {len(rr):8d} samples (ink {L[rr, cc].mean():.3f}, validation {V[rr, cc].mean():.3f}, holdout {hb.mean():.3f})", flush=True)
P = np.concatenate(P); LAB = np.concatenate(LAB); VAL = np.concatenate(VAL); HOLD = np.concatenate(HOLD); umL = a.um_label or 2.399
print(f"labels: {len(P)} samples, ink {LAB.mean():.3f}", flush=True)

# ---- coverage only from a vertex dump (an unflattened strip / wraps: no grid, no renders) ----
if a.vertices:
    D = np.fromfile(a.vertices, np.float32).reshape(-1, 5)[:, :3].astype(np.float64) * (a.vertex_um / umL)
    lo, hi = P.min(0) - 200, P.max(0) + 200; D = D[np.all((D >= lo) & (D <= hi), 1)]
    T = cKDTree(D); dd, ii = T.query(P, k=10, workers=-1)
    # the normal miss: the label point's distance to the plane of its 10 nearest vertices (PCA); the lateral vertex spacing (~33 um
    # in a 9.6 um wrap-dump) does not count. Points whose nearest vertex is farther than 120 um are misses at any tolerance.
    nb = D[ii]; c = nb.mean(1); X_ = nb - c[:, None, :]; C3 = np.einsum("nki,nkj->nij", X_, X_)
    w_, v_ = np.linalg.eigh(C3); nrm = v_[:, :, 0]
    miss_um = np.abs(np.einsum("ni,ni->n", P - c, nrm)) * umL; miss_um[dd[:, 0] * umL > 120] = np.inf
    print(f"vertices: {len(D)} near the labels; nearest vertex median {np.median(dd[:, 0]) * umL:.1f} um; normal miss median {np.median(miss_um[np.isfinite(miss_um)]):.1f} um", flush=True)
    rows = ["tol_um\tsplit\tlabelled\tcovered\tcoverage\tink_share\tmiss_med_um"]
    splits = {"all": np.ones(len(P), bool), "validation": VAL, "holdout": HOLD, "rest": ~VAL & ~HOLD}
    for t in tols:
        for nm, sp in splits.items():
            s_ = sp & (miss_um <= t); fm = miss_um[sp & np.isfinite(miss_um)]
            rows.append(f"{t:g}\t{nm}\t{sp.sum()}\t{s_.sum()}\t{s_.sum() / max(sp.sum(), 1):.3f}\t{LAB[s_].mean() if s_.any() else float('nan'):.3f}\t{np.median(fm) if len(fm) else float('nan'):.1f}")
    open(a.out, "w").write("\n".join(rows) + "\n"); print("\n".join(rows)); sys.exit(0)

# ---- our surfaces ----
lo, hi = P.min(0) - 200, P.max(0) + 200
verts, owner = [], []; S = []
for d in sorted(glob.glob(os.path.join(a.stripes, "*", "meta.json"))):
    d = os.path.dirname(d); m, X, Y, Z = load_xyz(d); f = float(m.get("voxel_size_um", umL)) / umL   # -> level-0 (label) voxels
    X, Y, Z = X * f, Y * f, Z * f; Z[Z <= 0] = -1
    k = len(S); S.append((os.path.basename(d), float(m["scale"][0]), X, Y, Z))
    sel = (Z > 0) & (Z >= lo[0]) & (Z <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]) & (X >= lo[2]) & (X <= hi[2])
    r, c = np.nonzero(sel)
    if len(r): verts.append(np.stack([Z[r, c], Y[r, c], X[r, c]], 1)); owner.append(np.stack([np.full(len(r), k), r, c], 1))
    print(f"wraps: {os.path.basename(d)} scale {m['scale'][0]} um {m.get('voxel_size_um')} -> x{f:.4f}: {len(r)} vertices near the labels", flush=True)
if not verts: sys.exit("no stripe vertex near the labelled meshes (coordinates / scale?)")
V3 = np.concatenate(verts); OW = np.concatenate(owner); T = cKDTree(V3)
dist0, idx = T.query(P, k=1, workers=-1)
k_, r_, c_ = OW[idx, 0], OW[idx, 1].astype(np.float64), OW[idx, 2].astype(np.float64)
# foot point on the bilinear patch (Gauss-Newton on (r, c), numerical Jacobian), per stripe
foot_r, foot_c, miss = r_.copy(), c_.copy(), dist0.copy()
for k in np.unique(k_):
    s = k_ == k; _, sc, X, Y, Z = S[k]; G = (Z, Y, X); r, c = r_[s], c_[s]; q = P[s]
    for _ in range(6):
        f0 = np.stack([bil(A, r, c) for A in G], 1); fr = np.stack([bil(A, r + .5, c) for A in G], 1) - f0; fc = np.stack([bil(A, r, c + .5) for A in G], 1) - f0
        Jm = np.stack([fr, fc], 2) * 2; e = q - f0
        JtJ = np.einsum("nki,nkj->nij", Jm, Jm) + 1e-6 * np.eye(2); Jte = np.einsum("nki,nk->ni", Jm, e)
        dlt = np.linalg.solve(JtJ, Jte[..., None])[..., 0]; dlt = np.clip(dlt, -1, 1)
        r = np.clip(r + dlt[:, 0], 0, Z.shape[0] - 1.001); c = np.clip(c + dlt[:, 1], 0, Z.shape[1] - 1.001)
    f0 = np.stack([bil(A, r, c) for A in G], 1); r0 = np.floor(r).astype(int).clip(0, Z.shape[0] - 2); c0 = np.floor(c).astype(int).clip(0, Z.shape[1] - 2)
    okc = (Z[r0, c0] > 0) & (Z[r0 + 1, c0] > 0) & (Z[r0, c0 + 1] > 0) & (Z[r0 + 1, c0 + 1] > 0)
    d = np.linalg.norm(q - f0, axis=1); d[~okc] = np.minimum(d[~okc], dist0[s][~okc])
    foot_r[s], foot_c[s], miss[s] = r, c, d
miss_um = miss * umL
print(f"miss: median {np.median(miss_um):.1f} um, p90 {np.percentile(miss_um, 90):.1f} um", flush=True)

# ---- the maps at the foot points ----
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
mapdir = a.map or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(a.stripes)))), "img_fill")
def find(nm, kind):
    c = [f"{nm}.tif", f"{nm}_ink.tif", f"{nm}_ink.jpg", f"{nm}.jpg"] if kind == "ink" else [f"{nm}_ct.tif", f"{nm}_ct.jpg"]
    for f in c:
        if os.path.exists(os.path.join(mapdir, f)): return os.path.join(mapdir, f)
    return None
INK = np.full(len(P), -1.0); CT = np.full(len(P), -1.0)
for k in np.unique(k_):
    nm, sc, X, Y, Z = S[k]; s = np.nonzero(k_ == k)[0]
    for kind, out in (("ink", INK), ("ct", CT)):
        f = find(nm, kind)
        if a.self_test or not f:
            if not f and not a.self_test: print(f"maps: no {kind} map for {nm} in {mapdir}", flush=True)
            continue
        im = np.asarray(Image.open(f).convert("L")) if not f.endswith(".tif") else tifffile.imread(f)
        cw, ch = Z.shape[1] / sc, Z.shape[0] / sc; fx, fy = im.shape[1] / cw, im.shape[0] / ch   # the map's px per canvas px
        if abs(fx - fy) > 0.02 * fx: print(f"maps: {f}: {im.shape[1]} x {im.shape[0]} is not proportional to the canvas {cw:.0f} x {ch:.0f}", flush=True); continue
        x = foot_c[s] / sc * fx; y = foot_r[s] / sc * fy   # (render pixel px <-> grid column px x scale: render_core.h rc_pixel)
        xi = np.clip(np.rint(x).astype(int), 0, im.shape[1] - 1); yi = np.clip(np.rint(y).astype(int), 0, im.shape[0] - 1)
        out[s] = im[yi, xi]
        print(f"maps: {nm} {kind} {os.path.basename(f)} ({fx:.3f} map px per canvas px): {len(s)} samples", flush=True)
if a.self_test: INK = LAB * 255.0; CT = np.random.default_rng(0).integers(0, 256, len(P)).astype(float)

def auc_f1(v, y):
    if len(v) == 0 or y.all() or not y.any(): return float("nan"), float("nan")
    b = np.clip(np.rint(v), 0, 255).astype(int); hi_ = np.bincount(b[y], minlength=256); hn = np.bincount(b[~y], minlength=256)
    tp = np.cumsum(hi_[::-1]); fp = np.cumsum(hn[::-1]); ti, tn = hi_.sum(), hn.sum()
    auc = (np.sum(hi_[::-1] * (np.concatenate([[0], fp[:-1]]) + 0.5 * hn[::-1]))) / (ti * tn); auc = 1 - auc
    pr = tp / np.maximum(tp + fp, 1); re = tp / ti; f1 = np.where(pr + re > 0, 2 * pr * re / np.maximum(pr + re, 1e-12), 0)
    return auc, f1.max()

rows = ["tol_um\tsplit\tlabelled\tcovered\tcoverage\tink_share\tink_auc\tink_f1best\tct_auc\tct_f1best\tmiss_med_um"]
splits = {"all": np.ones(len(P), bool), "validation": VAL, "holdout": HOLD, "rest": ~VAL & ~HOLD}
for t in tols:
    cov = miss_um <= t   # (the surface coverage needs no map: a run without renders gets it alone; the scores take the samples with a map value)
    for nm, sp in splits.items():
        s = sp & cov; si = s & (INK >= 0); sc_ = s & (CT >= 0)
        ia, if1 = auc_f1(INK[si], LAB[si]); ca, cf1 = auc_f1(CT[sc_], LAB[sc_])
        rows.append(f"{t:g}\t{nm}\t{sp.sum()}\t{s.sum()}\t{s.sum() / max(sp.sum(), 1):.3f}\t{LAB[s].mean() if s.any() else float('nan'):.3f}\t{ia:.4f}\t{if1:.4f}\t{ca:.4f}\t{cf1:.4f}\t{np.median(miss_um[sp]) if sp.any() else float('nan'):.1f}")
open(a.out, "w").write("\n".join(rows) + "\n"); print("\n".join(rows))

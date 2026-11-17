#!/usr/bin/env python3
"""pairab.py A.npz B.npz [--tol 40] [--block 2048] [--n 1000]: paired held-out ink AUC difference B - A of two wrapscore --dump runs on
the same stripes / labels (samples covered in both), with a bootstrap over spatial blocks (block^3 label voxels, default 2048 = ~4.9 mm
at 2.4 um: neighbouring samples are not independent). Prints AUC A, B, B - A and the 95 % interval."""
import argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=40)
ap.add_argument("--block", type=float, default=2048); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--split", default="heldout")
g = ap.parse_args(); A, B = np.load(g.a), np.load(g.b)
assert len(A["LAB"]) == len(B["LAB"]) and np.array_equal(A["LAB"], B["LAB"]), "different label samples"
sp = {"heldout": A["VAL"] | A["HOLD"], "validation": A["VAL"], "holdout": A["HOLD"], "rest": ~A["VAL"] & ~A["HOLD"]}[g.split]
m = sp & (A["miss_um"] <= g.tol) & (B["miss_um"] <= g.tol) & (A["INK"] >= 0) & (B["INK"] >= 0)
y, va, vb = A["LAB"][m].astype(bool), A["INK"][m].astype(np.float64), B["INK"][m].astype(np.float64)
def auc(v, y):   # rank AUC with ties at half
    from scipy.stats import rankdata
    r = rankdata(v); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
blk = np.floor(A["P"][m] / g.block).astype(np.int64); _, bid = np.unique(blk, axis=0, return_inverse=True); bid = bid.ravel(); nb = bid.max() + 1
idx = [np.flatnonzero(bid == k) for k in range(nb)]; rng = np.random.default_rng(1)
a0, b0 = auc(va, y), auc(vb, y); d = []
for _ in range(g.n):
    s = np.concatenate([idx[k] for k in rng.integers(0, nb, nb)]); d.append(auc(vb[s], y[s]) - auc(va[s], y[s]))
lo, hi = np.nanpercentile(d, [2.5, 97.5])
print(f"{g.split} tol {g.tol:g} um: n {m.sum()} ({nb} blocks), AUC A {a0:.4f} B {b0:.4f}, B - A {b0 - a0:+.4f} [{lo:+.4f}, {hi:+.4f}]")

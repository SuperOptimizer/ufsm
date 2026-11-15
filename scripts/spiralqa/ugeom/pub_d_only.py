#!/usr/bin/env python3
"""audit-ugeom: published -> ours distance only (the pub->ours half of surfcmp.py), for another wrap set, on the same published vertices
(vi) as an existing pub_dist.npz, so the per-vertex d / s / id can be compared stage to stage.
  pub_d_only.py PUB_TIFXYZ REF_NPZ OURS.f32 OUT.npz [--tan-um 60] [--workers 8]"""
import sys, numpy as np, tifffile
from scipy.spatial import cKDTree
pub, refz, ours, out = sys.argv[1:5]
tan_um = float(sys.argv[sys.argv.index('--tan-um') + 1]) if '--tan-um' in sys.argv else 60.0
W = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
um = 2.399
R = np.load(refz); vi = R['vi']; H, Wd = (int(v) for v in R['shape'])
X = tifffile.imread(f'{pub}/x.tif'); Y = tifffile.imread(f'{pub}/y.tif'); Z = tifffile.imread(f'{pub}/z.tif')
P = np.stack([Z, Y, X], -1); del X, Y, Z
du = np.zeros_like(P); du[:, 1:-1] = (P[:, 2:] - P[:, :-2]) * 0.5
dv = np.zeros_like(P); dv[1:-1] = (P[2:] - P[:-2]) * 0.5
n = np.cross(du.reshape(-1, 3)[vi], dv.reshape(-1, 3)[vi]); del du, dv
n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
Pv = P.reshape(-1, 3)[vi].astype(np.float64); del P
rec = np.fromfile(ours, dtype=np.float32).reshape(-1, 5)
qid = rec[:, 3].astype(np.int16); Q = rec[:, :3].astype(np.float64); del rec
print('ours', len(Q), flush=True)
t = cKDTree(Q, leafsize=32, balanced_tree=False, compact_nodes=False)
dpp, j = t.query(Pv, k=1, workers=W); del t
sg = np.einsum('ij,ij->i', Q[j] - Pv, n)       # sign arbitrary here (normal not oriented): only |s| used
tan = np.sqrt(np.maximum(dpp * dpp - sg * sg, 0))
d = np.where(tan * um < tan_um, np.abs(sg), dpp) * um
np.savez_compressed(out, d=d.astype(np.float32), id=qid[j])
print('written', out, 'd_med', np.median(d), 'cov50', (d < 50).mean(), flush=True)

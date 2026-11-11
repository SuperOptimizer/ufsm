"""Fibre tangent against the sheet normal of a band-phase raster (0..251 = turn x 252, 255 unknown): the normal is the gradient of the
phase (wrapped central differences, 3-voxel box mean), sampled at the points of a UFIB file. A fibre lies in its sheet, so |t . n| should
be near 0; a systematic value means the labelling's sheets are tilted against the fibres (or the fibre directions are off). analysis only.
usage: fibnormal.py FILE.fib PHASE.raw z,y,x,nz,ny,nx LEVEL"""
import sys, struct, numpy as np
from scipy.ndimage import uniform_filter
f = open(sys.argv[1], "rb"); assert f.read(4) == b"UFIB"; _, n, npts = struct.unpack("<iii", f.read(12)); off = np.frombuffer(f.read(4 * (n + 1)), "<i4"); fam = np.frombuffer(f.read(n), np.uint8); P = np.frombuffer(f.read(12 * npts), "<f4").reshape(-1, 3).astype(np.float64)
bz, by, bx, nz, ny, nx = map(int, sys.argv[3].split(",")); L = int(sys.argv[4]); ph = np.fromfile(sys.argv[2], np.uint8).reshape(nz, ny, nx)
known = ph <= 251; p = ph.astype(np.float32) / 252
g = []
for a in range(3):
    d = np.roll(p, -1, a) - np.roll(p, 1, a); d -= np.round(d); ok = known & np.roll(known, -1, a) & np.roll(known, 1, a); d[~ok] = 0; g.append(uniform_filter(d, 3) / 2)
okc = uniform_filter((known & np.roll(known, 1, 0) & np.roll(known, -1, 0) & np.roll(known, 1, 1) & np.roll(known, -1, 1) & np.roll(known, 1, 2) & np.roll(known, -1, 2)).astype(np.float32), 3) > 0.99
G = np.stack(g, -1); gn = np.linalg.norm(G, axis=-1)
T = np.zeros_like(P); F = np.zeros(npts, np.uint8)
for r in range(n): a, b = off[r], off[r + 1]; T[a:b] = np.gradient(P[a:b], axis=0); F[a:b] = fam[r]
T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9); q = np.rint((P + 0.5) / 2 ** L - 0.5 - [bz, by, bx]).astype(int); m = np.all((q >= 1) & (q < np.array([nz, ny, nx]) - 1), axis=1); q, T, F = q[m], T[m], F[m]
s = okc[q[:, 0], q[:, 1], q[:, 2]] & (gn[q[:, 0], q[:, 1], q[:, 2]] > 1e-3); q, T, F = q[s], T[s], F[s]; N = G[q[:, 0], q[:, 1], q[:, 2]]; pitch = 1 / np.linalg.norm(N, axis=1); N = N / np.linalg.norm(N, axis=1, keepdims=True)
c = np.abs((T * N).sum(1)); print(f"{len(c)} fibre points with a normal; sheet pitch from |grad phase|: median {np.median(pitch):.1f} level voxels; |n_z| median {np.median(np.abs(N[:, 0])):.3f}")
for k, nm in ((ord('H'), "H"), (ord('V'), "V")):
    v = c[F == k]
    if len(v): print(f"  {nm}: {len(v)} points, |t . n| mean {v.mean():.3f} median {np.median(v):.3f} p90 {np.quantile(v, 0.9):.3f}; angle out of the sheet median {np.degrees(np.arcsin(np.median(v))):.1f} deg; share > 0.5: {(v > 0.5).mean():.3f}")

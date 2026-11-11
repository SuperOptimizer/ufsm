"""Are the fibres of a UFIB file on the published prediction's fibres? Presence (level-4 chunks of the fiber-trace cache) at the fibre
points and at points shifted by a few voxels, plus |cos| between the fibre tangent and the decoded direction. analysis only.
usage: fibpres.py FILE.fib CACHE_PREFIX z0,z1 (level-4 chunk rows)"""
import sys, os, struct, numpy as np, blosc2
f = open(sys.argv[1], "rb"); assert f.read(4) == b"UFIB"; _, n, npts = struct.unpack("<iii", f.read(12)); off = np.frombuffer(f.read(4 * (n + 1)), "<i4"); fam = f.read(n); P = np.frombuffer(f.read(12 * npts), "<f4").reshape(-1, 3).astype(np.float64)
pre = sys.argv[2]; z0, z1 = map(int, sys.argv[3].split(","))
T = np.zeros_like(P)
for r in range(n): a, b = off[r], off[r + 1]; T[a:b] = np.gradient(P[a:b], axis=0) if b - a > 1 else 0
T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9); q = (P + 0.5) / 16 - 0.5; m = (q[:, 0] >= z0 * 64) & (q[:, 0] < z1 * 64); q, T = q[m], T[m]
cache = {}
def chunk(k, c):
    key = (k, c)
    if key not in cache:
        p = f"{pre}_{k}.ome.zarr/4/{c[0]}/{c[1]}/{c[2]}"; cache[key] = np.frombuffer(blosc2.decompress(open(p, "rb").read()), np.uint8).reshape(64, 64, 64) if os.path.exists(p) and os.path.getsize(p) else None
    return cache[key]
def samp(k, pts):
    i = np.rint(pts).astype(int); out = np.full(len(i), -1.0); cs = i // 64
    for c in {tuple(x) for x in cs}:
        a = chunk(k, c); s = np.all(cs == c, axis=1)
        if a is not None: j = i[s] - np.array(c) * 64; out[s] = a[j[:, 0], j[:, 1], j[:, 2]]
    return out
pr = samp("presence", q); ok = pr >= 0; print(f"{len(q)} points in the z range, {ok.sum()} in cached chunks; presence mean {pr[ok].mean():.1f}, share >= 64: {(pr[ok] >= 64).mean():.3f}")
for d in (2, 4):
    v = [samp("presence", q + np.array(s) * d) for s in ((0, d and 1, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1))]; v = np.concatenate([x[x >= 0] for x in v]); print(f"   shifted {d} level-4 voxels in y / x: presence mean {v.mean():.1f}, share >= 64: {(v >= 64).mean():.3f}")
u = (samp("nx", q) - 128) / 127; w = (samp("ny", q) - 128) / 127; s = ok & (pr >= 64); zz = np.sqrt(np.clip(1 - u * u - w * w, 0, 1)); cs = np.abs(T[:, 0] * zz + T[:, 1] * w + T[:, 2] * u)[s]
print(f"   |cos(tangent, predicted axis)| at presence >= 64: mean {cs.mean():.3f} median {np.median(cs):.3f} ({s.sum()} points)")

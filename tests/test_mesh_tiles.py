"""mesh-tiles: the T-voxel level-L tiles a mesh touches (every valid vertex + the axis-aligned box of each valid right / down edge),
grown by --dilate tiles, clipped to --shape: the exact set against a numpy reference, on a random curved tifxyz with invalid vertices
and a coarse grid step (edges longer than a tile: the edge boxes, not the vertices alone, must mark the tiles between)."""
import pathlib
import subprocess
import tempfile

import importlib.util
import sys

_miss = [m for m in ("numpy", "tifffile") if importlib.util.find_spec(m) is None]
if _miss:
    print(f"SKIP: no python {', '.join(_miss)} (the reference needs it)")
    sys.exit(0)

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout
    return r.stdout


def ref(X, Y, Z, ts, D, nt):
    V = (Z > 0) & (X >= 0) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    P = np.stack([Z, Y, X], -1).astype(np.float64)
    s = set()
    H, W = X.shape
    for r in range(H):
        for c in range(W):
            if not V[r, c]:
                continue
            for b in ((r, c), (r, c + 1), (r + 1, c)):
                if b[0] >= H or b[1] >= W or not V[b]:
                    continue
                lo = np.floor(np.minimum(P[r, c], P[b]) / ts).astype(int); hi = np.floor(np.maximum(P[r, c], P[b]) / ts).astype(int)
                for z in range(lo[0], hi[0] + 1):
                    for y in range(lo[1], hi[1] + 1):
                        for x in range(lo[2], hi[2] + 1):
                            if min(z, y, x) >= 0:
                                s.add((z, y, x))
    g = set()
    for (z, y, x) in s:
        for dz in range(-D, D + 1):
            for dy in range(-D, D + 1):
                for dx in range(-D, D + 1):
                    q = (z + dz, y + dy, x + dx)
                    if min(q) >= 0 and all(q[i] < nt[i] for i in range(3)):
                        g.add(q)
    return sorted(g)


rng = np.random.default_rng(7)
H, W = 23, 31
u, v = np.meshgrid(np.arange(W), np.arange(H))
X = (100 + 90 * u + 40 * np.sin(v / 3)).astype(np.float32)          # steps of 90 level-0 voxels > the 64-voxel tile below
Y = (500 + 300 * np.cos(u / 9) + 7 * v).astype(np.float32)
Z = (30 + 85 * v + 20 * np.sin(u / 4)).astype(np.float32)
bad = rng.random((H, W)) < 0.15
X[bad] = -1; Y[bad] = -1; Z[bad] = -1
with tempfile.TemporaryDirectory(prefix="ufsm-meshtiles-") as tmp:
    d = pathlib.Path(tmp); m = d / "m"; m.mkdir()
    for n, a in (("x", X), ("y", Y), ("z", Z)):
        t.imwrite(m / f"{n}.tif", a)
    (m / "meta.json").write_text('{"format": "tifxyz", "type": "seg", "uuid": "m", "scale": [1, 1]}')
    for L, T, D, shape in ((0, 64, 0, None), (1, 32, 1, None), (2, 16, 1, (1500, 900, 2500)), (0, 512, 0, None)):
        ts = T << L
        nt = [10 ** 9] * 3 if shape is None else [-(-s // ts) for s in shape]
        o = d / f"t{L}{T}{D}.txt"
        a = ["--level", L, "--tile", T, "--dilate", D] + ([] if shape is None else ["--shape", ",".join(map(str, shape))])
        out = run(BIN, "mesh-tiles", o, *a, m)
        got = [tuple(int(w) for w in ln.split()) for ln in o.read_text().splitlines()]
        want = ref(X, Y, Z, ts, D, nt)
        assert got == want, f"L {L} T {T} D {D}: {len(got)} tiles vs reference {len(want)}"
        print(f"L {L} T {T} D {D} shape {shape}: {len(got)} tiles == reference ({out.strip()})")
print("mesh-tiles: OK")

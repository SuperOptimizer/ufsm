"""Which encoding do the published nx / ny fibre-direction channels use? Paris4_v1 streamline tangents (the uploader traced them through
this prediction) against candidate decodings in single level-4 chunks. analysis only."""
import sys, numpy as np, blosc2
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
from fiblib import Map, open_afv, runs_in_box
d = sys.argv[1]; db, _ = open_afv("/vesuvius/ufsm/fibers/raw/Paris4_v1.afv"); mp = Map(4)
for c in sys.argv[2:]:
    cz, cy, cx = map(int, c.split("_")); a = {}
    for k in ("presence", "nx", "ny"): a[k] = np.frombuffer(blosc2.decompress(open(f"{d}/{k}_{c}", "rb").read()), np.uint8).reshape(64, 64, 64)
    box = (cz * 64, cy * 64, cx * 64, 64, 64, 64); runs = runs_in_box(db, mp, box, step=1.0)
    P = np.concatenate([r[3] for r in runs]); T = np.concatenate([r[4] for r in runs]); q = np.rint(P - box[:3]).astype(int); ok = np.all((q >= 0) & (q < 64), 1); q = q[ok]; T = T[ok]
    pr = a["presence"][q[:, 0], q[:, 1], q[:, 2]]; u = (a["nx"][q[:, 0], q[:, 1], q[:, 2]].astype(float) - 128) / 127; v = (a["ny"][q[:, 0], q[:, 1], q[:, 2]].astype(float) - 128) / 127
    print(f"chunk {c}: {len(runs)} runs, {len(q)} samples, presence at fibre samples mean {pr.mean():.0f} (chunk mean {a['presence'].mean():.0f}), share >= 64: {(pr >= 64).mean():.3f}; |(u, v)| mean {np.hypot(u, v).mean():.3f} max {np.hypot(u, v).max():.3f}")
    w = np.sqrt(np.clip(1 - u * u - v * v, 0, 1)); tz, ty, tx = T[:, 0], T[:, 1], T[:, 2]; m = pr >= 64
    cand = {"hemisphere (x, y, +z)": (w, v, u), "hemisphere (y, x, +z)": (w, u, v), "hemisphere (x=u, z=v, +y)": (v, w, u), "hemisphere(z=u, y=v, +x)": (u, v, w)}
    for n, (dz, dy, dx) in cand.items():
        cs = np.abs(tz * dz + ty * dy + tx * dx)[m]; print(f"   {n:28s} |cos| mean {cs.mean():.3f} median {np.median(cs):.3f}")
    # double-angle in the xy plane? angle of (u, v) vs 2 * atan2(ty, tx)
    da = np.angle(np.exp(1j * (np.arctan2(v, u) - 2 * np.arctan2(ty, tx))))[m]; print(f"   double angle (cos 2a, sin 2a): |d| median {np.median(np.abs(da)):.3f} rad; tz^2 vs 1 - |uv|^2 corr {np.corrcoef(tz[m] ** 2, 1 - (u * u + v * v)[m])[0, 1]:.3f}")

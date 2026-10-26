#!/usr/bin/env python3
"""ufsm ink3d-ext (C/CUDA, fp16 storage) against the published gambiarra 3D ink network (tests/gambiarra_ref.py, fp32 CPU torch).

1. --raw-in parity: two random 256^3 parents and two real PHercParis4 2.4 um parents (20260411134726 level 0) in one batch: the
   p1 / p99 normalisation limits (np.percentile) and the 64^3 logits (max / mean |d|).
2. Box geometry: ufsm ink3d-ext over a small level-0 box (--out-um 4.8,9.6) against a numpy re-implementation of villa volume.py
   (global crop grid, Gaussian blend, sigmoid, 2^3 mean pooling, prob254) fed with the C logits of the same crops (--raw-in), so
   only the tiling / blend / encoding is under test here.

CT: UFSM_GAMBIARRA_CT root (default the fitkit p4box copy) and key; skips (exit 0) when torch, the weights, the CT, the GPU or
build/ufsm are missing."""
import json, math, os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
CT_ROOT = pathlib.Path(os.environ.get("UFSM_GAMBIARRA_CT", "~/ufsm-data/fitkit/p4box/ct")).expanduser()
CT_KEY = os.environ.get("UFSM_GAMBIARRA_KEY", "PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/0")
# level-0 parent origins (z, y, x) of the real tiles (papyrus inside the local fitkit shards 47-50 / 15-18 / 13-17)
REAL = [tuple(int(v) for v in s.split(",")) for s in os.environ.get("UFSM_GAMBIARRA_AT", "49536,16512,15104;50048,17024,14592").split(";")]
BOX = [int(v) for v in os.environ.get("UFSM_GAMBIARRA_BOX", "49600,16600,15200,160,200,120").split(",")]


def skip(msg):
    print(f"SKIP: {msg}"); sys.exit(0)


def read(z, y, x, n, out):
    r = subprocess.run([str(UFSM), "read", str(CT_ROOT), CT_KEY, str(z), str(y), str(x), str(n), str(n), str(n), str(out)], capture_output=True, text=True)
    if r.returncode:
        skip(f"CT read failed: {r.stderr[-300:]}")
    import numpy as np
    return np.fromfile(out, np.uint8).reshape(n, n, n)


def grid_axes(size, patch=64):
    """villa volume.grid_axes (Python round = half to even)"""
    import numpy as np
    count = math.ceil((int(size) - patch) / (patch // 2)) + 1
    s = np.array([round((int(size) - patch) / (count - 1) * i) if count > 1 else 0 for i in range(count)], np.int64)
    return s[(s >= 0) & (s + patch <= size)]


def raw(parents, td, name):
    import numpy as np
    p = td / f"{name}.u8"
    np.stack(parents).astype(np.uint8).tofile(p)
    out = td / name
    r = subprocess.run([str(UFSM), "ink3d-ext", "--weights", str(W), "--raw-in", str(p), "--raw-n", str(len(parents)), "--raw-out", str(out)], capture_output=True, text=True)
    print(r.stderr.strip())
    if r.returncode:
        print("FAIL: ufsm ink3d-ext --raw-in failed"); sys.exit(1)
    lg = np.fromfile(out / "logits.f32", np.float32).reshape(len(parents), 64, 64, 64)
    st = np.array([[float(v) for v in l.split()] for l in (out / "stats.txt").read_text().split("\n") if l.strip()])
    return lg, st


def main():
    global W
    try:
        import numpy as np, torch
    except ImportError:
        skip("torch / numpy missing")
    import gambiarra_ref as R
    W = R.WEIGHTS
    for p in [W, UFSM, CT_ROOT / CT_KEY / "zarr.json"]:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        skip("no GPU")
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("UFSM_TEST_TMP", str(pathlib.Path.home() / "ufsm-data"))) as td:
        td = pathlib.Path(td)
        # 1. parity
        rng = np.random.default_rng(7)
        par = [rng.integers(0, 256, (256,) * 3, dtype=np.uint8),
               np.clip(rng.normal(110, 40, (256,) * 3), 0, 255).astype(np.uint8)]
        par += [read(z, y, x, 256, td / f"p{i}.u8") for i, (z, y, x) in enumerate(REAL)]
        lg, st = raw(par, td, "parity")
        net = R.build()
        lo, hi, ref = R.run_parents(net, np.stack(par))
        for i, name in enumerate(["random uniform", "random normal"] + [f"real {z},{y},{x}" for z, y, x in REAL]):
            d = np.abs(lg[i] - ref[i])
            ok_st = st[i, 0] == lo[i] and st[i, 1] == hi[i]
            print(f"{name}: p1/p99 C {st[i, 0]:g}/{st[i, 1]:g} ref {lo[i]:g}/{hi[i]:g} {'equal' if ok_st else 'DIFFER'}; "
                  f"logits |d| max {d.max():.4f} mean {d.mean():.5f} (ref range {ref[i].min():.2f} .. {ref[i].max():.2f}, CT mean {par[i].mean():.1f})")
            # measured 2026-10-26 (fp16 vs fp32): real max 0.010 / 0.026, mean 0.0009 / 0.0015; random (logits to -32) max 0.06 / 0.12,
            # mean 0.004 / 0.006
            real = i >= 2
            fails += (not ok_st) + (d.max() > (0.06 if real else 0.25)) + (d.mean() > (0.005 if real else 0.012))
        # 2. box geometry
        meta = json.loads((CT_ROOT / CT_KEY / "zarr.json").read_text())
        V = meta["shape"]
        b0 = [v // 4 * 4 for v in BOX[:3]]
        b1 = [-(-(BOX[a] + BOX[3 + a]) // 4) * 4 for a in range(3)]
        r48 = [(b0[a] // 2, b1[a] // 2) for a in range(3)]
        axes = []
        for a in range(3):
            g = grid_axes(-(-V[a] // 2))
            g = g[(2 * g - 64 >= 0) & (2 * g + 192 <= V[a])]
            axes.append([int(o) for o in g if o < r48[a][1] and o + 64 > r48[a][0]])
        crops = [(oz, oy, ox) for oz in axes[0] for oy in axes[1] for ox in axes[2]]
        print(f"box {BOX}: 4.8 region {r48}, {len(crops)} crops")
        lo_ = [min(c[a] for c in crops) * 2 - 64 for a in range(3)]
        hi_ = [max(c[a] for c in crops) * 2 + 192 for a in range(3)]
        n = [hi_[a] - lo_[a] for a in range(3)]
        rr = subprocess.run([str(UFSM), "read", str(CT_ROOT), CT_KEY, *map(str, lo_), *map(str, n), str(td / "blk.u8")], capture_output=True, text=True)
        if rr.returncode:
            skip(f"CT read failed: {rr.stderr[-300:]}")
        blk = np.fromfile(td / "blk.u8", np.uint8).reshape(n)
        pars = [blk[2 * oz - 64 - lo_[0]:2 * oz + 192 - lo_[0], 2 * oy - 64 - lo_[1]:2 * oy + 192 - lo_[1], 2 * ox - 64 - lo_[2]:2 * ox + 192 - lo_[2]] for oz, oy, ox in crops]
        clg, _ = raw(pars, td, "boxcrops")
        g1 = np.exp(-0.5 * (np.arange(64) - 32.0) ** 2 / 64.0)
        g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
        sh = [r48[a][1] - r48[a][0] for a in range(3)]
        S = np.zeros(sh); Wt = np.zeros(sh)
        for (oz, oy, ox), l, p in zip(crops, clg, pars):
            if p[64:192, 64:192, 64:192].max() < 25:
                continue
            o = [oz - r48[0][0], oy - r48[1][0], ox - r48[2][0]]
            sl_dst, sl_src = [], []
            for a in range(3):
                a0, a1 = max(o[a], 0), min(o[a] + 64, sh[a])
                sl_dst.append(slice(a0, a1)); sl_src.append(slice(a0 - o[a], a1 - o[a]))
            S[tuple(sl_dst)] += (g * l)[tuple(sl_src)]; Wt[tuple(sl_dst)] += g[tuple(sl_src)]
        known = Wt > 0
        p = np.where(known, 1 / (1 + np.exp(-np.where(known, S / np.maximum(Wt, 1e-30), 0))), 0)
        e48 = np.where(known, np.rint(p * 254), 255).astype(np.uint8)
        pz = p.reshape(sh[0] // 2, 2, sh[1] // 2, 2, sh[2] // 2, 2); kz = known.reshape(pz.shape)
        nk = kz.sum((1, 3, 5)); e96 = np.where(nk > 0, np.rint((pz * kz).sum((1, 3, 5)) / np.maximum(nk, 1) * 254), 255).astype(np.uint8)
        out = td / "box"
        r = subprocess.run([str(UFSM), "ink3d-ext", "--weights", str(W), "--ct", str(CT_ROOT), CT_KEY, "--box", ",".join(map(str, BOX)), "--out", str(out),
                            "--out-um", "9.6,4.8", "--batch", "5", "--band", "64"], capture_output=True, text=True)
        print(r.stderr.strip()[-1500:])
        if r.returncode:
            print("FAIL: ufsm ink3d-ext box run failed"); return 1
        for lv, e, f in [("4.8", e48, 2), ("9.6", e96, 4)]:
            o = [b0[a] // f for a in range(3)]
            rr = subprocess.run([str(UFSM), "read", str(out), lv, *map(str, o), *map(str, e.shape), str(td / f"o{lv}.u8")], capture_output=True, text=True)
            if rr.returncode:
                print(f"FAIL: cannot read {lv}: {rr.stderr[-300:]}"); return 1
            c = np.fromfile(td / f"o{lv}.u8", np.uint8).reshape(e.shape)
            d = np.abs(c.astype(int) - e.astype(int))
            unk = (c == 255) != (e == 255)
            print(f"box {lv} um {e.shape}: known {np.mean(e != 255):.3f}, unknown mismatches {unk.sum()}, prob254 |d| max {d[(c != 255) & (e != 255)].max(initial=0)} "
                  f"mean {d[(c != 255) & (e != 255)].mean():.4f}")
            fails += unk.sum() > 0 or d[(c != 255) & (e != 255)].max(initial=0) > 1
            # just outside the box: unknown
            o2 = [o[0] - 2, o[1], o[2]]
            rr = subprocess.run([str(UFSM), "read", str(out), lv, *map(str, o2), "2", str(e.shape[1]), str(e.shape[2]), str(td / "edge.u8")], capture_output=True, text=True)
            ed = np.fromfile(td / "edge.u8", np.uint8)
            print(f"  rows above the box: {np.mean(ed == 255):.3f} unknown")
            fails += not (ed == 255).all()
    print("FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

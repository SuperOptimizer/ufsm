#!/usr/bin/env python3
"""ufsm dinovol-embed --raw-in (C/CUDA fp16) against the official dinovol PyTorch model (tests/dinovol_ref.py, fp32 CPU) on real CT tiles:
patch tokens (x_norm_patchtokens), the CNN skips, the FPN grid, and the dense / ink unit embeddings at points. Skips (exit 0) when
torch, the checkpoint, the official sources, the CT, the GPU or build/ufsm are missing.

CT: a 9.6 um Paris 4 crop (UFSM_DINOVOL_CT root, UFSM_DINOVOL_KEY, default the fitkit p4box level 2). Tiles 128 (and 192 with
UFSM_DINOVOL_FULL=1). Bounds (fp16 vs fp32, measured 2026-10-21 + margin): token cosine >= 0.999 mean / 0.99 min, dense / ink cosine
>= 0.995 mean / 0.95 min."""
import os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dinovol_ref as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
CT_ROOT = pathlib.Path(os.environ.get("UFSM_DINOVOL_CT", "~/ufsm-data/fitkit/p4box/ct")).expanduser()
CT_KEY = os.environ.get("UFSM_DINOVOL_KEY", "PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")


def skip(msg):
    print(f"SKIP: {msg}"); sys.exit(0)


def cos_rows(a, b):
    import numpy as np
    a = a.reshape(-1, a.shape[-1]).astype(np.float64); b = b.reshape(-1, b.shape[-1]).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def main():
    try:
        import numpy as np, torch
    except ImportError:
        skip("torch / numpy missing")
    for p in [R.CKPT, R.DATA / "src/dinovol_2/model/dinov2_eva.py", R.DATA / "pylib/timm", UFSM, CT_ROOT / CT_KEY]:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        skip("no GPU")
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))
    bb, head = R.build()
    tiles = [128] + ([192] if os.environ.get("UFSM_DINOVOL_FULL") == "1" else [])
    rng = np.random.default_rng(1)
    center, scale, hi = 84.0, 31.5, 200.0
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("UFSM_TEST_TMP", str(pathlib.Path.home() / "ufsm-data"))) as td:
        td = pathlib.Path(td)
        for T in tiles:
            raw = td / f"ct{T}.u8"
            r = subprocess.run([str(UFSM), "read", str(CT_ROOT), CT_KEY, "12400", "4300", "3900", str(T), str(T), str(T), str(raw)], capture_output=True, text=True)
            if r.returncode:
                skip(f"CT read failed: {r.stderr[-300:]}")
            u = np.fromfile(raw, np.uint8).reshape(T, T, T)
            pts = np.concatenate([rng.uniform(-0.5, T - 0.5, size=(200, 3)), rng.integers(0, T, size=(200, 3)) + 0.0,
                                  np.array([[0, 0, 0], [T - 1, T - 1, T - 1], [3.5, 3.5, 3.5]])]).astype(np.float32)
            pts.tofile(td / "pts.f32")
            out = td / f"out{T}"; out.mkdir()
            r = subprocess.run([str(UFSM), "dinovol-embed", "--ckpt", str(R.CKPT), "--raw-in", str(raw), "--raw-out", str(out), "--tile", str(T),
                                "--norm", f"{center},{scale},{hi}", "--points", str(td / "pts.f32")], capture_output=True, text=True)
            print(r.stderr.strip())
            if r.returncode:
                print("FAIL: ufsm dinovol-embed failed"); return 1
            ref = R.run_tile(bb, head, u, center, scale, hi, points=pts)
            g = T // 8
            tok = np.fromfile(out / "tokens.f32", np.float32).reshape(g, g, g, -1)
            c = cos_rows(tok, ref["tokens"]); err = np.abs(tok - ref["tokens"])
            print(f"T {T} tokens: cos mean {c.mean():.6f} min {c.min():.6f}; |d| max {err.max():.4f} mean {err.mean():.5f} (|ref| max {np.abs(ref['tokens']).max():.2f})")
            fails += c.mean() < 0.99999 or c.min() < 0.9999
            for k in range(3):
                s = ref["skips"][k]; e = s.shape[0]
                mine = np.fromfile(out / f"skip{k}.f32", np.float32).reshape(s.shape)
                c = cos_rows(mine, s); d = np.abs(mine - s)
                print(f"  skip{k} {s.shape}: cos mean {c.mean():.6f} min {c.min():.6f}; |d| max {d.max():.4f} mean {d.mean():.5f}")
                fails += c.mean() < 0.9999
            f = ref["fpn"]; mine = np.fromfile(out / "fpn.f32", np.float32).reshape(f.shape)
            c = cos_rows(mine, f); print(f"  fpn {f.shape}: cos mean {c.mean():.6f} min {c.min():.6f}; |d| max {np.abs(mine - f).max():.4f}")
            fails += c.mean() < 0.9999
            di = np.fromfile(out / "dense_ink.f32", np.float32).reshape(len(pts), -1)
            nd = ref["dense"].shape[1]
            for name, a, b in [("dense", di[:, :nd], ref["dense"]), ("ink", di[:, nd:], ref["ink"])]:
                c = cos_rows(a, b)
                print(f"  {name}: cos mean {c.mean():.6f} min {c.min():.6f}; |d| max {np.abs(a - b).max():.4f}")
                fails += c.mean() < 0.9999 or c.min() < 0.999
    print("FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

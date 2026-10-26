#!/usr/bin/env python3
"""ufsm hercuvit --raw-in (C/CUDA, fp16 storage) against the authors' HercuViT v0 PyTorch model (tests/hercuvit_ref.py, fp32 CPU)
on one real 128^3 CT window: the per-window robust normalisation statistics, the encoder tokens, the surface probability
(sigmoid of the head's logit margin), the final mask embeddings, and the per-voxel instance owner (argmax over the 64 queries) /
membership. Skips (exit 0) when torch, the checkpoint, the vendored sources, the CT, the GPU or build/ufsm are missing.

CT: a 9.6 um Paris 4 crop (UFSM_HERCUVIT_CT root, UFSM_HERCUVIT_KEY; default the fitkit p4box level 2, as tests/test_dinovol.py).
UFSM_HERCUVIT_AT z,y,x moves the window. Bounds (fp16 vs fp32, measured 2026-10-26 + margin) are printed with the numbers."""
import os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hercuvit_ref as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
CT_ROOT = pathlib.Path(os.environ.get("UFSM_HERCUVIT_CT", "~/ufsm-data/fitkit/p4box/ct")).expanduser()
CT_KEY = os.environ.get("UFSM_HERCUVIT_KEY", "PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
AT = [int(v) for v in os.environ.get("UFSM_HERCUVIT_AT", "12400,4300,3900").split(",")]


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
    for p in [R.CKPT, R.DATA / "src/hercunet/hercuvit/netbuild.py", R.DV / "pylib/timm", UFSM, CT_ROOT / CT_KEY]:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        skip("no GPU")
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))
    T = 128
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("UFSM_TEST_TMP", str(pathlib.Path.home() / "ufsm-data"))) as td:
        td = pathlib.Path(td)
        raw = td / "win.u8"
        r = subprocess.run([str(UFSM), "read", str(CT_ROOT), CT_KEY, *map(str, AT), str(T), str(T), str(T), str(raw)], capture_output=True, text=True)
        if r.returncode:
            skip(f"CT read failed: {r.stderr[-300:]}")
        u = np.fromfile(raw, np.uint8).reshape(T, T, T)
        out = td / "out"
        if os.environ.get("UFSM_HERCUVIT_OUT"):   # a --raw-out dir of this window computed earlier (only that step needs the GPU)
            out = pathlib.Path(os.environ["UFSM_HERCUVIT_OUT"])
        else:
            r = subprocess.run([str(UFSM), "hercuvit", "--ckpt", str(R.CKPT), "--raw-in", str(raw), "--raw-out", str(out), "--window", str(T)],
                               capture_output=True, text=True)
            print(r.stderr.strip())
            if r.returncode:
                print("FAIL: ufsm hercuvit failed"); return 1
        net = R.build()
        ref = R.run_window(net, u)
        st = [float(v) for v in (out / "stats.txt").read_text().split()]
        rs = R.robust_stats(u)
        ds = max(abs(a - b) for a, b in zip(st, rs))
        print(f"robust stats (lo, hi, median, scale): C {st} ref {list(rs)}; max |d| {ds:.2e}")
        fails += ds > 1e-3
        g = T // 8
        tok = np.fromfile(out / "tokens.f32", np.float32).reshape(g, g, g, -1)
        c = cos_rows(tok, ref["tokens"])
        print(f"tokens: cos mean {c.mean():.6f} min {c.min():.6f}; |d| max {np.abs(tok - ref['tokens']).max():.4f} (|ref| max {np.abs(ref['tokens']).max():.2f})")
        fails += c.mean() < 0.9999 or c.min() < 0.999
        m = np.fromfile(out / "margin.f32", np.float32).reshape(T, T, T)
        p, pr = 1 / (1 + np.exp(-m.astype(np.float64))), 1 / (1 + np.exp(-ref["margin"].astype(np.float64)))
        dp = np.abs(p - pr)
        sp, srf = p > 0.4, pr > 0.4
        dice = 2 * (sp & srf).sum() / max(1, sp.sum() + srf.sum())
        print(f"surface prob: max |d| {dp.max():.4f} mean {dp.mean():.6f} p99.99 {np.quantile(dp, 0.9999):.4f}; margin |d| max {np.abs(m - ref['margin']).max():.3f}; "
              f"p > 0.4: C {sp.mean():.4f} ref {srf.mean():.4f} dice {dice:.5f}")
        fails += dp.max() > 0.05 or dice < 0.995
        me = np.fromfile(out / "embed.f32", np.float32).reshape(-1, 32)
        c = cos_rows(me, ref["embed"])
        print(f"mask embeddings (64 x 32): cos mean {c.mean():.6f} min {c.min():.6f}; |d| max {np.abs(me - ref['embed']).max():.4f}")
        fails += c.min() < 0.99
        ow = np.fromfile(out / "owner.u8", np.uint8).reshape(T, T, T)
        mb = np.fromfile(out / "memb.f32", np.float32).reshape(T, T, T)
        agree_all = (ow == ref["owner"]).mean(); gate = srf
        agree_s = (ow[gate] == ref["owner"][gate]).mean() if gate.any() else 1.0
        conf = gate & (ref["memb"] > 0.7)
        agree_c = (ow[conf] == ref["owner"][conf]).mean() if conf.any() else 1.0
        print(f"instance owner agreement: all voxels {agree_all:.5f}, surface (ref p > 0.4) {agree_s:.5f}, surface with ref membership > 0.7 {agree_c:.5f}; "
              f"membership |d| max {np.abs(mb - ref['memb']).max():.4f} mean {np.abs(mb - ref['memb']).mean():.5f}; queries used (surface): C {len(np.unique(ow[gate]))} ref {len(np.unique(ref['owner'][gate]))}")
        fails += agree_s < 0.98 or agree_c < 0.995
    print("FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

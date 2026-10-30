#!/usr/bin/env python3
"""ufsm villawind --raw-in (C/CUDA: fp16 storage, fp32 accumulation and residual stream) against the authors' villa winding model
(tests/villawind_ref.py, fp32 CPU) on one real Paris 4 slab [T, T, L] (a 9.6 um CT box read as the slab; the network does not care
about the geometry) with an invalid corner, so the validity channel and the masked standardisation are exercised. Compares every
encoder stage, the trunk, each axial attention block, the decoder, the head and the per-voxel phase (monotone cumsum along the ray),
and the turn counts between ray samples 64 voxels apart. Skips (exit 0) when torch, the checkpoint, the villa source, the CT, the GPU
or build/ufsm are missing. UFSM_VILLAWIND_T / _L shrink the slab (default 128 x 384, the model's recipe)."""
import os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import villawind_ref as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
CT_ROOT = pathlib.Path(os.environ.get("UFSM_VILLAWIND_CT", "~/ufsm-data/fitkit/p4box/ct")).expanduser()
CT_KEY = os.environ.get("UFSM_VILLAWIND_KEY", "PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
AT = [int(v) for v in os.environ.get("UFSM_VILLAWIND_AT", "12400,4300,3900").split(",")]


def skip(msg):
    print(f"SKIP: {msg}"); sys.exit(0)


def main():
    try:
        import numpy as np, torch
    except ImportError:
        skip("torch / numpy missing")
    for p in [R.CKPT, R.SRC, UFSM, CT_ROOT / CT_KEY]:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        skip("no GPU")
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))
    T, L = int(os.environ.get("UFSM_VILLAWIND_T", "128")), int(os.environ.get("UFSM_VILLAWIND_L", "384"))
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("UFSM_TEST_TMP", str(pathlib.Path.home() / "ufsm-data"))) as td:
        td = pathlib.Path(td)
        raw = td / "box.u8"
        r = subprocess.run([str(UFSM), "read", str(CT_ROOT), CT_KEY, *map(str, AT), str(T), str(T), str(L), str(raw)], capture_output=True, text=True)
        if r.returncode:
            skip(f"CT read failed: {r.stderr[-300:]}")
        im = np.fromfile(raw, np.uint8).reshape(T, T, L)
        va = np.ones_like(im, dtype=np.uint8)
        va[: T // 4, : T // 4, : L // 6] = 0          # an invalid corner (the model zeroes it and sees valid = 0)
        im = im * va
        (td / "in").mkdir()
        im.tofile(td / "in/image.u8"); va.tofile(td / "in/valid.u8")
        out = td / "out"
        if os.environ.get("UFSM_VILLAWIND_OUT"):   # a --raw-out dir of this slab computed earlier (only that step needs the GPU)
            out = pathlib.Path(os.environ["UFSM_VILLAWIND_OUT"]); r = None
        else:
          r = subprocess.run([str(UFSM), "villawind", "--ckpt", str(R.CKPT), "--raw-in", str(td / "in"), "--raw-out", str(out), "--T", str(T), "--L", str(L), "--reps", "3"],
                           capture_output=True, text=True)
          print(r.stderr.strip())
          if r.returncode:
            print("FAIL: ufsm villawind failed"); return 1
        net = R.build()
        ref = R.run_slab(net, im, va.astype(bool))
        t8, h2 = T // 8, T // 2
        shapes = {"stage0": (h2, h2, L, 32), "stage1": (T // 4, T // 4, L, 64), "stage2": (t8, t8, L, 96), "trunk_in": (t8, t8, L, 192),
                  "stem": (t8, t8, L, 192), **{f"block{k}": (t8, t8, L, 192) for k in range(6)}, "decoder": (T // 4, T // 4, L, 96),
                  "head0": (h2, h2, L, 32), "logits": (4, h2, h2, L), "phase": (T, T, L)}
        for name, shp in shapes.items():
            c = np.fromfile(out / f"{name}.f32", np.float32).reshape(shp); rf = ref[name].reshape(shp)
            d = np.abs(c.astype(np.float64) - rf); rel = d.max() / (np.abs(rf).max() + 1e-30)
            print(f"{name:9s} |d| max {d.max():.5f} mean {d.mean():.6f}  |ref| max {np.abs(rf).max():.3f}  rel {rel:.2e}")
            fails += rel > (5e-3 if name != "phase" else 2e-3)
        ph, pr = np.fromfile(out / "phase.f32", np.float32).reshape(T, T, L), ref["phase"]
        dt = (ph[:, :, 64:] - ph[:, :, :-64]) - (pr[:, :, 64:] - pr[:, :, :-64])
        print(f"turns over 64 ray voxels: |d| max {np.abs(dt).max():.4f} mean {np.abs(dt).mean():.5f} p99.9 {np.quantile(np.abs(dt), 0.999):.4f} "
              f"(ref turns mean {(pr[:, :, 64:] - pr[:, :, :-64]).mean():.3f}); whole ray |d| max {np.abs(ph[:, :, -1] - pr[:, :, -1]).max():.4f} of {pr[:, :, -1].mean():.2f}")
        fails += np.abs(dt).max() > 0.05
    print("PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

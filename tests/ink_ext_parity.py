"""Shared parity harness of tests/test_ink_ext_*.py: the official PyTorch forward of an external ink network (fp32, CPU) against
ufsm ink-ext --raw-in (fp16 tensor cores) on the same uint8 tiles. Skips (exit 0) when torch, the weights, the official
sources, the GPU or build/ufsm are missing.

Data (UFSM_INKEXT, default ~/ufsm-data/inkext): weights/{v8in.safetensors, hecate_2.4um.pth, hecate_9.6um.pth, reader-v3.pth}
and the official sources src/v8in (HF YoussefMoNader/ink-8um-v8in), src/hecate (HF scrollprize/hecate), src/villa (ScrollPrize/
villa merge-ink-pipelines: ink-detection/koine_machines, vesuvius/src/vesuvius/models + image_proc). Real tiles are CT boxes of
the local PHerc1299 pyramid (UFSM_INKEXT_CT root + key, default ~/ufsm-data/vol PHerc1299/...), skipped when absent."""
import os, pathlib, subprocess, sys, tempfile, types

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
DATA = pathlib.Path(os.environ.get("UFSM_INKEXT", "~/ufsm-data/inkext")).expanduser()
CT_ROOT = pathlib.Path(os.environ.get("UFSM_INKEXT_CT_ROOT", "~/ufsm-data/vol")).expanduser()
CT_KEY = os.environ.get("UFSM_INKEXT_CT_KEY", "PHerc1299/volumes/20260309130042-2.399um-0.2m-78keV-masked.zarr")


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(0)


def need(paths):
    try:
        import numpy, torch  # noqa: F401
    except ImportError:
        skip("torch / numpy not installed")
    for p in paths:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if not UFSM.exists():
        skip("build/ufsm missing")
    try:
        gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except OSError:
        gpus = ""
    if "GPU" not in gpus:
        skip("no GPU")
    import torch
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))


def stub_hf_hub():
    """ink8um/modeling.py subclasses huggingface_hub.PyTorchModelHubMixin (only for from_pretrained); a stand-in when absent."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        m = types.ModuleType("huggingface_hub")

        class PyTorchModelHubMixin:
            def __init_subclass__(cls, **kw):
                super().__init_subclass__()
        m.PyTorchModelHubMixin = PyTorchModelHubMixin
        sys.modules["huggingface_hub"] = m


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def random_tiles(n, D, T, seed):
    """uniform noise, a smooth random field and a layered sheet-like pattern, uint8"""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        if k % 3 == 0:
            t = rng.integers(0, 256, (D, T, T))
        elif k % 3 == 1:
            f = rng.normal(size=(D // 2 + 2, T // 8 + 2, T // 8 + 2))
            z = np.linspace(0, f.shape[0] - 1.001, D); y = np.linspace(0, f.shape[1] - 1.001, T)
            iz, iy = z.astype(int), y.astype(int)
            g = f[iz][:, iy][:, :, iy]
            t = 110 + 45 * g + rng.normal(scale=8, size=(D, T, T))
        else:
            zz = np.arange(D)[:, None, None]
            yy, xx = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
            sheet = np.exp(-((zz - D / 2 - 2 * np.sin(xx / 17.0)) / (D / 6)) ** 2)
            ink = (np.sin(xx / 5.0) * np.sin(yy / 7.0) > 0.6)[None]
            t = 30 + 140 * sheet * (1 + 0.3 * ink) + rng.normal(scale=10, size=(D, T, T))
        out.append(np.clip(t, 0, 255).astype(np.uint8))
    return np.stack(out)


def real_tiles(n, D, T, level, origins):
    """CT boxes (D x T x T) of the local PHerc1299 pyramid level as stand-in surface volumes; None when unavailable"""
    import numpy as np
    arr = CT_ROOT / CT_KEY / str(level)
    if not arr.exists():
        return None
    out = []
    with tempfile.TemporaryDirectory(prefix="ufsm-inkext-") as d:
        for k in range(n):
            z, y, x = origins[k % len(origins)]
            raw = pathlib.Path(d) / "t.raw"
            r = subprocess.run([UFSM, "read", CT_ROOT, f"{CT_KEY}/{level}", str(z), str(y), str(x), str(D), str(T), str(T), raw], capture_output=True, text=True)
            if r.returncode:
                return None
            t = np.fromfile(raw, np.uint8).reshape(D, T, T)
            if (t > 0).mean() < 0.5:
                return None
            out.append(t)
    return np.stack(out)


def run_c(model, weights, tiles, batch=2):
    import numpy as np
    n, D, T, _ = tiles.shape
    with tempfile.TemporaryDirectory(prefix="ufsm-inkext-") as d:
        i, o = pathlib.Path(d) / "in.u8", pathlib.Path(d) / "out.f32"
        tiles.astype(np.uint8).tofile(i)
        r = subprocess.run([UFSM, "ink-ext", "--model", model, "--weights", weights, "--raw-in", i, "--raw-out", o, "--batch", str(batch)], capture_output=True, text=True, timeout=1200)
        sys.stderr.write(r.stderr)
        assert r.returncode == 0, f"ufsm ink-ext {model} failed"
        return np.fromfile(o, np.float32).reshape(n, T, T)


def compare(name, ref, got, atol, rtol=0.02, mean_tol=None):
    """max |logit error| <= atol + rtol |ref| per pixel; returns the max abs error"""
    import numpy as np
    ref = np.asarray(ref, np.float64); got = np.asarray(got, np.float64)
    assert ref.shape == got.shape, (ref.shape, got.shape)
    err = np.abs(ref - got)
    bound = atol + rtol * np.abs(ref)
    worst = float((err / bound).max())
    pr = 1 / (1 + np.exp(-ref)); pg = 1 / (1 + np.exp(-got))
    print(f"{name}: max |dlogit| {err.max():.4g}  mean {err.mean():.3g}  logit range [{ref.min():.2f}, {ref.max():.2f}]  max |dprob| {np.abs(pr - pg).max():.3g}  worst/bound {worst:.3f}")
    assert np.isfinite(got).all(), f"{name}: non-finite C output"
    assert worst <= 1.0, f"{name}: logit error above tolerance"
    if mean_tol is not None:
        assert err.mean() <= mean_tol, f"{name}: mean logit error {err.mean():.4g} > {mean_tol}"
    return float(err.max())

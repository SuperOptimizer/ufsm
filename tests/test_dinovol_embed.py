#!/usr/bin/env python3
"""ufsm dinovol-embed (lattice tiling, blending, masks, sharded zarr v3 output, read back by ufsm dinovol-read) against the official
eval/embed_volume.py RegionEmbedder.run driving the official PyTorch model (fp32 CPU) on the same CT region: tile 128 (margin = ramp = 16,
stride 80), 1 x 2 x 2 lattice cells, dense stride 2, float16 storage, fixed normalisation. vc_reader (the official reader / transforms) is
replaced by stand-ins: a reader of `ufsm read` boxes and the "fixed" normalisation (min(x, hi) - center) / scale. Skips (exit 0) when
torch, the checkpoint, the official sources, the CT, the GPU or build/ufsm are missing.
Bounds (fp16 vs fp32): tokens / dense / ink cosine mean >= 0.9999, min >= 0.999; identical zero (masked) sets."""
import contextlib, os, pathlib, subprocess, sys, tempfile, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dinovol_ref as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
CT_ROOT = pathlib.Path(os.environ.get("UFSM_DINOVOL_CT", "~/ufsm-data/fitkit/p4box/ct")).expanduser()
CT_KEY = os.environ.get("UFSM_DINOVOL_KEY", "PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
T, MARGIN, RAMP, DS = 128, 16, 16, 2
CENTER, SCALE, HI = 84.0, 31.5, 200.0
BOX = (12400, 4320, 3920, 80, 160, 160)   # 1 x 2 x 2 cells of the stride-80 lattice


def skip(msg):
    print(f"SKIP: {msg}"); sys.exit(0)


def stubs(td):
    import numpy as np, torch
    vc = types.ModuleType("vc_reader"); vc.__path__ = []
    tr = types.ModuleType("vc_reader.transforms")

    class Pipeline:
        def __init__(self, *a, **k): pass

        def batch(self, sample, seeds=None, device=None):
            x = sample["image"].double()
            return {"image": ((torch.clamp(x, max=HI) - CENTER) / SCALE).float()}
    tr.Pipeline = Pipeline; tr.Image = lambda **k: None; tr.Normalize = lambda *a, **k: None
    vc.transforms = tr
    nc = types.ModuleType("numcodecs"); nc.Zstd = lambda **k: None; sys.modules["numcodecs"] = nc
    tq = types.ModuleType("tqdm"); tq.tqdm = lambda *a, **k: types.SimpleNamespace(update=lambda *a: None, close=lambda: None) if not a else a[0]; sys.modules.setdefault("tqdm", tq)
    sys.modules["vc_reader"] = vc; sys.modules["vc_reader.transforms"] = tr
    for name, attrs in [("dinovol_2.dataset.cuda_pipeline", {"NORMALIZATION_SCHEMES": ("robust",), "normalization_transforms": lambda *a, **k: []}),
                        ("dinovol_2.dataset.volume_reader", {"open_volume": None})]:
        m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m

    def configure(backbone, shape):
        R.configure(backbone, shape[0])
    nv = types.ModuleType("dinovol_2.eval.napari_visualizer")
    nv.LoadedBackbone = object; nv._configure_backbone_for_spatial_shape = configure; nv.load_backbone_from_checkpoint = None
    sys.modules["dinovol_2.eval.napari_visualizer"] = nv


class Reader:
    """vc_reader-like region reads through `ufsm read` (zero outside the volume)."""
    def __init__(self, td):
        import numpy as np
        r = subprocess.run([str(UFSM), "info", str(CT_ROOT), CT_KEY], capture_output=True, text=True).stdout
        s = r.split("shape")[1].split("shard")[0].replace("x", " ").split()
        self.shape = tuple(int(v) for v in s); self.td = td; self.dtype = np.uint8

    def read_region(self, start, shape):
        import numpy as np
        out = pathlib.Path(self.td) / "region.u8"
        subprocess.run([str(UFSM), "read", str(CT_ROOT), CT_KEY, *[str(int(v)) for v in start], *[str(int(v)) for v in shape], str(out)], check=True, capture_output=True)
        return np.fromfile(out, np.uint8).reshape(shape)


class Sink:
    def __init__(self): self.arrays = {}

    def write(self, name, start, block):
        self.arrays.setdefault(name, []).append((tuple(start), block.astype("float32")))


def main():
    try:
        import numpy as np, torch
    except ImportError:
        skip("torch / numpy missing")
    for p in [R.CKPT, R.DATA / "src/dinovol_2/eval/embed_volume.py", R.DATA / "pylib/timm", UFSM, CT_ROOT / CT_KEY]:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        skip("no GPU")
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))
    bb, head = R.build()
    with tempfile.TemporaryDirectory(dir=os.environ.get("UFSM_TEST_TMP", str(pathlib.Path.home() / "ufsm-data"))) as td:
        stubs(td)
        import dinovol_2.eval.embed_volume as E
        E.torch.autocast = lambda *a, **k: contextlib.nullcontext()
        loaded = types.SimpleNamespace(backbone=bb, dense_head=head, device=torch.device("cpu"), patch_size=(8, 8, 8), embedding_type="deeper",
                                       global_input_size=(T, T, T))
        lat = [E.AxisLattice(T, MARGIN, RAMP)] * 3
        emb = E.RegionEmbedder(Reader(td), loaded, transforms=[], outputs=["tokens", "dense", "ink"], dense_stride=DS, lattice=lat, dtype="float16")
        sink = Sink()
        z0, y0, x0, nz, ny, nx = BOX
        stats = emb.run(sink, z_range=(z0, z0 + nz), y_range=(y0, y0 + ny), x_range=(x0, x0 + nx), progress=False)
        print("reference run:", stats)
        out = pathlib.Path(td) / "emb"
        r = subprocess.run([str(UFSM), "dinovol-embed", str(CT_ROOT), CT_KEY, str(out), "--ckpt", str(R.CKPT), "--box", ",".join(map(str, BOX)),
                            "--tile", str(T), "--margin", str(MARGIN), "--ramp", str(RAMP), "--dense-stride", str(DS), "--dtype", "float16",
                            "--norm", f"{CENTER},{SCALE},{HI}"], capture_output=True, text=True)
        print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
        if r.returncode:
            print(r.stderr[-2000:]); print("FAIL: dinovol-embed"); return 1
        fails = 0
        for name, step in [("tokens", 8), ("dense", DS), ("ink", DS)]:
            for (zi, yi, xi), ref in sink.arrays[name]:
                gz = z0 // step + zi   # the reference arrays start at z_range[0]; ours span the volume
                f = pathlib.Path(td) / "r.f32"
                subprocess.run([str(UFSM), "dinovol-read", str(out), name, f"{gz},{yi},{xi},{ref.shape[0]},{ref.shape[1]},{ref.shape[2]}", str(f)], check=True)
                mine = np.fromfile(f, np.float32).reshape(ref.shape)
                a = mine.reshape(-1, ref.shape[-1]).astype(np.float64); b = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
                za, zb = np.abs(a).sum(1) == 0, np.abs(b).sum(1) == 0
                live = ~za & ~zb
                c = (a[live] * b[live]).sum(1) / (np.linalg.norm(a[live], axis=1) * np.linalg.norm(b[live], axis=1))
                print(f"{name} block {ref.shape}: {live.sum()} vectors, zero sets equal {bool((za == zb).all())}, cos mean {c.mean():.6f} min {c.min():.6f}, "
                      f"|d| max {np.abs(a - b).max():.4f}")
                fails += (za != zb).any() or c.mean() < 0.9999 or c.min() < 0.999
    print("FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

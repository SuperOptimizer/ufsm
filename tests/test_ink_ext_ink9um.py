#!/usr/bin/env python3
"""ufsm ink-ext --model ink9um against the official scrollprize/ink_9um network (hybrid_3d2d seed42 / seed43 step-075000,
model_type vesuvius_unet_3d_stem_2d) built by villa's koine_machines make_model + vesuvius NetworkFromConfig
(merge-ink-pipelines), fp32 on the CPU: 17 x 128 x 128 uint8 tiles through vesuvius normalize_robust (p1 / p99 clip, median /
MAD) as koine infer.py feeds them (its center_crop_layer_indices: layers 2..18 of a 21-layer / 6..22 of a 28-layer volume);
the 128 x 128 ink logits.

Tiles: random patterns, a half-empty edge tile, and REAL surface-volume tiles = raw chunks (28 x 128 x 128, uncompressed) of the
published PHerc0139 w039 native 9.362 um surface volume (S3 .../20260302000000-w039_2026030210/surface-volumes/
9.362um-1.2m-113keV-volume-20250728140407.zarr/0/0/<cy>/<cx>) in $UFSM_INK9UM/parity/c_<cy>_<cx>.u8, the model's own input.
Weights: $UFSM_INK9UM (default ~/ufsm-data/inkft/teachers/ink9um)/seed{42,43}/step-075000.pth (HF scrollprize/ink_9um)."""
import os, pathlib, sys, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D9 = pathlib.Path(os.environ.get("UFSM_INK9UM", "~/ufsm-data/inkft/teachers/ink9um")).expanduser()
W42, W43 = D9 / "seed42/step-075000.pth", D9 / "seed43/step-075000.pth"
V = P.DATA / "src/villa"
VES, KOI = V / "vesuvius/src/vesuvius", V / "ink-detection/koine_machines"
P.need([W42, VES / "models/build/build_network_from_config.py", VES / "image_proc/intensity/normalization.py", KOI / "models/make_model.py", KOI / "models/hybrid_3d2d.py"])
import numpy as np, torch


def pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
    return m


def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m


class _Unused:
    def __init__(self, *a, **k):
        raise RuntimeError("not part of ink_9um")


for name, path in [("vesuvius", VES), ("vesuvius.models", VES / "models"), ("vesuvius.models.build", VES / "models/build"),
                   ("vesuvius.models.utilities", VES / "models/utilities"), ("vesuvius.image_proc", VES / "image_proc"),
                   ("vesuvius.image_proc.intensity", VES / "image_proc/intensity"), ("koine_machines", KOI), ("koine_machines.models", KOI / "models")]:
    pkg(name, path)
stub("vesuvius.models.build.primus_wrapper", PrimusEncoder=_Unused, PrimusDecoder=_Unused)
pkg("vesuvius.models.build.pretrained_backbones", VES / "models/build/pretrained_backbones")
stub("vesuvius.models.build.pretrained_backbones.dinov2", build_dinov2_backbone=_Unused, build_dinov2_decoder=_Unused)
stub("koine_machines.models.resnet3d", ResNet3DSegmentationModel=_Unused)
from koine_machines.models.make_model import make_model  # noqa: E402
from vesuvius.image_proc.intensity.normalization import normalize_robust  # noqa: E402


def build(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    cfg["crop_size"] = cfg["patch_size"] = [int(v) for v in cfg["patch_size"]]   # infer.py normalize_training_config_for_inference
    cfg.setdefault("targets", {}).setdefault("ink", {})["out_channels"] = 1
    assert cfg["model_type"] == "vesuvius_unet_3d_stem_2d" and cfg["patch_size"] == [17, 128, 128], cfg["patch_size"]
    sd = payload.get("ema_model") or payload["model"]   # infer.py prefer_ema
    model = make_model(cfg).eval()
    model.load_state_dict(sd, strict=True)
    return model


def forward(model, tiles):
    out = []
    with torch.inference_mode():
        for t in tiles:
            x = torch.from_numpy(normalize_robust(t.astype(np.float32)))[None, None]
            out.append(model(x)["ink"].numpy()[:, 0])
    return np.concatenate(out)


tiles = P.random_tiles(3, 17, 128, seed=13)
tiles[2, :, 70:, :] = 0   # a tile edge beyond the image (zero padded, as koine reads edge blocks)
real = []
for f in sorted((D9 / "parity").glob("c_*.u8")):
    a = np.fromfile(f, dtype=np.uint8)
    if a.size == 28 * 128 * 128:
        real.append(a.reshape(28, 128, 128)[6:23])   # center_crop_layer_indices(28 -> 17): 28 // 2 - 17 // 2 = 6
real = np.stack(real) if real else None
if real is None:
    print("no real surface-volume tiles (parity/c_*.u8): random tiles only")
allt = tiles if real is None else np.concatenate([tiles, real])
for tag, w in [("seed42", W42), ("seed43", W43)]:
    if not w.exists():
        print(f"{w} missing: {tag} skipped"); continue
    ref = forward(build(w), allt)
    got = P.run_c("ink9um", w, allt, batch=3)
    P.compare(f"ink9um {tag} random", ref[:3], got[:3], atol=0.1, mean_tol=0.02)
    if real is not None:
        P.compare(f"ink9um {tag} real surface volume", ref[3:], got[3:], atol=0.1, mean_tol=0.02)
        pr, pg = 1 / (1 + np.exp(-ref[3:])), 1 / (1 + np.exp(-got[3:]))
        print(f"ink9um {tag} real: prob max |dp| {np.abs(pr - pg).max():.5f}, ref prob range {pr.min():.3f}..{pr.max():.3f}")


# ---- end to end: ink-ext --layers (a 28-layer surface-volume stack) against koine infer.py's tiling + blend, with the torch model ----
import struct, subprocess, tempfile, zlib  # noqa: E402


def write_tif(path, a):   # uint8 single strip, uncompressed
    h, w = a.shape
    tags = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8 + 2 + 12 * 9 + 4),
            (277, 3, 1, 1), (278, 3, 1, h), (279, 4, 1, w * h)]
    with open(path, "wb") as f:
        f.write(b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(tags)))
        for t, ty, n, v in tags:
            f.write(struct.pack("<HHI", t, ty, n) + (struct.pack("<HH", v, 0) if ty == 3 else struct.pack("<I", v)))
        f.write(struct.pack("<I", 0)); f.write(a.astype(np.uint8).tobytes())


def read_tif(path):   # tiffw's uint8 strips (deflate or none)
    b = open(path, "rb").read(); off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; tg = {}
    for i in range(n):
        t, ty, c = struct.unpack_from("<HHI", b, off + 2 + 12 * i); v = off + 2 + 12 * i + 8
        vals = [struct.unpack_from("<H" if ty == 3 else "<I", b, v)[0]] if c == 1 else list(struct.unpack_from(f"<{c}I", b, struct.unpack_from("<I", b, v)[0]))
        tg[t] = vals
    w, h = tg[256][0], tg[257][0]; data = b""
    for o, l in zip(tg[273], tg[279]):
        s = b[o:o + l]; data += zlib.decompress(s) if tg[259][0] == 8 else s
    return np.frombuffer(data, np.uint8)[: w * h].reshape(h, w)


def positions(L, p, s):   # infer.py _sliding_positions_1d
    if L <= p:
        return [0]
    v = list(range(0, max(1, L - p + 1), s))
    if v[-1] != L - p:
        v.append(L - p)
    return v


def koine_blend(model, stack, shifts):   # infer.py: center_crop_layer_indices, 128 px, stride 64, hann (floor 1e-3), empty skipped
    Ds, H, Wd = stack.shape; T = 128
    w1 = np.hanning(T); win = np.outer(w1, w1); win = np.maximum(win / win.max(), 1e-3)
    acc = np.zeros((H, Wd)); n = 0
    for s in shifts:
        z0 = Ds // 2 - 17 // 2 + s
        num = np.zeros((H, Wd)); den = np.zeros((H, Wd))
        for y in positions(H, T, 64):
            for x in positions(Wd, T, 64):
                t = np.zeros((17, T, T), np.uint8); blk = stack[z0:z0 + 17, y:y + T, x:x + T]; t[:, :blk.shape[1], :blk.shape[2]] = blk
                if not t.any():
                    continue
                p = 1 / (1 + np.exp(-forward(model, t[None])[0].astype(np.float64)))
                h, w = blk.shape[1], blk.shape[2]
                num[y:y + h, x:x + w] += p[:h, :w] * win[:h, :w]; den[y:y + h, x:x + w] += win[:h, :w]
        acc += np.where(den > 1e-6, num / np.maximum(den, 1e-12), 0); n += 1
    return acc / n


chunks = {f.name: np.fromfile(f, np.uint8).reshape(28, 128, 128) for f in (D9 / "parity").glob("c_*.u8") if f.stat().st_size == 28 * 128 * 128}
if len(chunks) >= 4:
    c = [chunks[k] for k in sorted(chunks)][:4]
    stack = np.concatenate([np.concatenate([c[0], c[1]], 2), np.concatenate([c[2], c[3]], 2)], 1)[:, :230, :200].copy()
    stack[:, 150:, 140:] = 0   # an empty corner (koine skips all-zero patches)
    model = build(W42)
    with tempfile.TemporaryDirectory(prefix="ufsm-ink9um-") as d:
        d = pathlib.Path(d); (d / "layers").mkdir()
        for i in range(stack.shape[0]):
            write_tif(d / "layers" / f"{i:02d}.tif", stack[i])
        for tag, extra, shifts in [("plain", [], [0]), ("zwin", ["--shift", "1", "--zwin", "-1,0,1"], [0, 1, 2])]:
            r = subprocess.run([P.UFSM, "ink-ext", "--model", "ink9um", "--weights", W42, "--layers", d / "layers", "--out", d / tag, "--name", "t", "--batch", "4", *extra],
                               capture_output=True, text=True, timeout=600)
            sys.stderr.write(r.stderr)
            assert r.returncode == 0, f"ink-ext --layers ({tag}) failed"
            out = read_tif(d / tag / "t_ink9um.tif").astype(np.int32)
            ref = np.rint(255 * koine_blend(model, stack, shifts)).astype(np.int32)
            err = np.abs(out - ref)
            print(f"ink9um --layers {tag} (shifts {shifts}): {out.shape}, max |out - koine blend| {err.max()} / 255, mean {err.mean():.3f}")
            assert out.shape == ref.shape and err.max() <= 3, f"ink9um {tag}: blend differs from infer.py"
else:
    print("fewer than 4 surface-volume chunks: --layers blend check skipped")
print("test_ink_ext_ink9um: OK")

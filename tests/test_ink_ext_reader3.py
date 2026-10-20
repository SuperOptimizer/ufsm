#!/usr/bin/env python3
"""ufsm ink-ext --model reader3 against the official Reader V3 network (domenicor046/reader-v3, model_type
vesuvius_unet_3d_stem_2d) built by villa's koine_machines make_model + vesuvius NetworkFromConfig (merge-ink-pipelines), fp32
on the CPU: 17 x 384 x 384 uint8 tiles through vesuvius normalize_robust (p1 / p99 clip, median / MAD) as koine infer.py feeds
them; the 384 x 384 ink logits. Modules the network does not use (Primus, DINOv2 backbones, the ResNet3D segmentation model,
the vesuvius / koine package __init__s) are stubbed so only torch and numpy are needed."""
import pathlib, sys, types
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

W = P.DATA / "weights/reader-v3.pth"
V = P.DATA / "src/villa"
VES, KOI = V / "vesuvius/src/vesuvius", V / "ink-detection/koine_machines"
P.need([W, VES / "models/build/build_network_from_config.py", VES / "image_proc/intensity/normalization.py", KOI / "models/make_model.py", KOI / "models/hybrid_3d2d.py"])
import numpy as np, torch


def pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
    return m


def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m


class _Unused:
    def __init__(self, *a, **k):
        raise RuntimeError("not part of Reader V3")


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

payload = torch.load(W, map_location="cpu", weights_only=True)
cfg = dict(payload["config"])
cfg["crop_size"] = cfg["patch_size"] = [int(v) for v in cfg["patch_size"]]   # infer.py normalize_training_config_for_inference
cfg.setdefault("targets", {}).setdefault("ink", {})["out_channels"] = 1
model = make_model(cfg).eval()
model.load_state_dict(payload["model"], strict=True)

tiles = P.random_tiles(3, 17, 384, seed=11)
tiles[2, :, 200:, :] = 0   # a tile edge beyond the image (zero padded, as koine reads edge blocks)
real = P.real_tiles(2, 17, 384, 2, [(5000, 2000, 2000), (5800, 1900, 2400)])
if real is not None:
    tiles = np.concatenate([tiles, real])
else:
    print("no local CT: random tiles only")
refs = []
with torch.inference_mode():
    for t in tiles:
        x = torch.from_numpy(normalize_robust(t.astype(np.float32)))[None, None]
        refs.append(model(x)["ink"].numpy()[:, 0])
ref = np.concatenate(refs)
got = P.run_c("reader3", W, tiles, batch=2)
P.compare("reader3 random", ref[:3], got[:3], atol=0.1, mean_tol=0.02)
if real is not None:
    P.compare("reader3 real CT", ref[3:], got[3:], atol=0.1, mean_tol=0.02)
print("test_ink_ext_reader3: OK")

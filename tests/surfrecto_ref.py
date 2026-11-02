"""Reference forward of the recto-surface U-Net (HF scrollprize/surface_recto_3dunet, checkpoint_inference_ready.pth) built by the
villa vesuvius NetworkFromConfig itself (github ScrollPrize/villa @fd97a4b, sparse clone of vesuvius/src/vesuvius/models under
UFSM_SURFRECTO/src/villa), fp32 on the CPU, for tests/test_surfrecto.py only (never part of a pipeline). The vesuvius package
__init__ (data loaders: nrrd, zarr, ...) is bypassed with namespace stubs; only models/build + models/utilities are imported.

Data dir UFSM_SURFRECTO (default ~/ufsm-data/surfrecto): model/checkpoint_inference_ready.pth, src/villa.
run_window(u8 [P,P,P]) mirrors vesuvius.models.run.inference on one patch: instance_zscore (float32 (x - mean) / max(std, 1e-8)),
the model, margin = logits[:, 1] - logits[:, 0] (2-class softmax p = sigmoid(margin)). With keep=True the encoder stage outputs and
decoder stage outputs are returned too (forward hooks)."""
import os, pathlib, sys, types

DATA = pathlib.Path(os.environ.get("UFSM_SURFRECTO", "~/ufsm-data/surfrecto")).expanduser()
CKPT = DATA / "model" / "checkpoint_inference_ready.pth"
SRC = DATA / "src" / "villa" / "vesuvius" / "src"


def setup():
    for n, sub in [("vesuvius", ""), ("vesuvius.models", "models"), ("vesuvius.models.build", "models/build"),
                   ("vesuvius.models.utilities", "models/utilities")]:
        if n in sys.modules: continue
        m = types.ModuleType(n); m.__path__ = [str(SRC / "vesuvius" / sub)]; sys.modules[n] = m
    # architectures this checkpoint does not use (Primus ViT: timm / einops; MedNeXt): stubbed so only the U-Net path is imported
    for n, names in [("vesuvius.models.build.primus_wrapper", ["PrimusEncoder", "PrimusDecoder"]),
                     ("vesuvius.models.build.mednext", []), ("vesuvius.models.build.mednext.factory", ["get_mednext_v1_config"]),
                     ("vesuvius.models.build.mednext_wrapper", ["MedNeXtEncoder", "MedNeXtDecoder"]),
                     ("vesuvius.models.build.pretrained_backbones", []),
                     ("vesuvius.models.build.pretrained_backbones.dinov2", ["build_dinov2_backbone", "build_dinov2_decoder"])]:
        if n in sys.modules: continue
        m = types.ModuleType(n); m.__path__ = []
        for c in names: setattr(m, c, type(c, (), {}))
        sys.modules[n] = m


def build():
    import torch
    setup()
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = ck["model_config"]

    class Mgr:   # vesuvius.models.run.inference MinimalConfigManager
        def __init__(self, c):
            self.model_config = c; self.targets = c.get("targets", {})
            self.train_patch_size = c.get("train_patch_size", c.get("patch_size", (128, 128, 128)))
            self.train_batch_size = c.get("train_batch_size", c.get("batch_size", 2)); self.in_channels = c.get("in_channels", 1)
            self.autoconfigure = c.get("autoconfigure", False); self.enable_deep_supervision = bool(c.get("enable_deep_supervision", False))
            self.model_name = c.get("model_name", "Model"); self.spacing = [1] * len(self.train_patch_size)

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        net = NetworkFromConfig(Mgr(cfg))
    missing, unexpected = net.load_state_dict(ck["model"], strict=True), None
    net.eval()
    return net, ck


def zscore(u8):
    import numpy as np
    x = u8.astype(np.float32)
    return (x - np.mean(x)) / max(np.std(x), 1e-8)


def run_window(net, u8, keep=False):
    import numpy as np, torch
    x = torch.from_numpy(zscore(u8))[None, None]
    feats = {}
    hooks = []
    if keep:
        for s, st in enumerate(net.shared_encoder.stages):
            hooks.append(st.register_forward_hook(lambda m, i, o, s=s: feats.__setitem__(f"enc{s}", o.detach()[0].permute(1, 2, 3, 0).numpy().copy())))
        dec = net.task_decoders["surface"]
        for s, st in enumerate(dec.stages):
            hooks.append(st.register_forward_hook(lambda m, i, o, s=s: feats.__setitem__(f"dec{s}", o.detach()[0].permute(1, 2, 3, 0).numpy().copy())))
    with torch.no_grad():
        out = net(x)
    for h in hooks: h.remove()
    lg = out["surface"] if isinstance(out, dict) else out
    lg = lg[0].numpy()
    return (lg[1] - lg[0]).astype(np.float32), feats

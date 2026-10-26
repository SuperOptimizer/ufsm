"""Reference forward of HercuViT v0 (HF jimmylomro/hercuvit-v0, checkpoint_latest.pth) built from the authors' sources
(github jimmylomro/hercUNet branch hercuvit de4acb7, the hercunet/hercuvit subpackage vendored under UFSM_HERCUVIT/src), fp32 on
the CPU, for tests/test_hercuvit.py only (never part of a pipeline). Data dir UFSM_HERCUVIT (default ~/ufsm-data/hercuvit):
checkpoint_latest.pth, src/hercunet. timm + einops from the dinovol test's pylib (UFSM_DINOVOL/pylib); timm's torchvision import is
stubbed as in tests/dinovol_ref.py.

run_window(u8 [T,T,T]) mirrors hercuvit/infer/model.py forward_meshes on one window: dinovol_normalise (villa normalize_robust),
the net with run_instance=True, surface margin = sem_logits[0][:, 1] - [:, 0], mask logits = last mask_embed . mask_features,
owner = argmax, membership = exp(max - logsumexp)."""
import os, pathlib, sys, types

DATA = pathlib.Path(os.environ.get("UFSM_HERCUVIT", "~/ufsm-data/hercuvit")).expanduser()
DV = pathlib.Path(os.environ.get("UFSM_DINOVOL", "~/ufsm-data/dinovol")).expanduser()
CKPT = DATA / "checkpoint_latest.pth"


def setup():
    lib = DV / "pylib"
    sys.path.insert(0, str(DATA / "src"))
    sys.path.insert(0, str(lib))
    for n in ["torchvision", "torchvision.ops", "torchvision.ops.misc"]:
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    sys.modules["torchvision.ops.misc"].FrozenBatchNorm2d = type("FrozenBatchNorm2d", (), {})
    m = types.ModuleType("timm"); m.__path__ = [str(lib / "timm")]; sys.modules["timm"] = m


def build():
    """the net exactly as hercuvit/infer/loader.load_hercuvit builds it (recipe fields from the checkpoint)"""
    import torch
    setup()
    from hercunet.hercuvit import netbuild
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    r = ck["recipe"]
    net = netbuild.build_from_cfg(
        ck["model_cfg"], device="cpu", attention_backend="auto", arch_override=ck.get("arch_override") or {},
        global_blocks=tuple(range(1, int(ck["model_cfg"]["depth"]) + 1)), window_size=tuple(ck["window_size"]),
        n_sem_classes=r["n_sem_classes"], mask_dim=r["mask_dim"], n_queries=r["n_queries"], query_dim=r["query_dim"],
        instance_layers=r["instance_layers"], deep_supervision=r["deep_supervision"], with_instance=True, decoder=r["decoder"],
        neck_heads=r["neck_heads"], add_null_query=r["add_null_query"], learned_queries=r["learned_queries"],
        fps_queries=r["fps_queries"], cross_attn=r["cross_attn"])
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    return net


def robust_stats(u8):
    """normalize_robust's (lower, upper, median, scaled MAD) as the reference computes them"""
    import numpy as np
    arr = u8.astype(np.float32).reshape(-1)
    lo = float(np.percentile(arr, 1.0)); hi = float(np.percentile(arr, 99.0))
    arr = np.clip(arr, lo, hi)
    med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
    return lo, hi, med, 1.4826 * mad


def run_window(net, u8):
    import numpy as np, torch
    from hercunet.hercuvit.data.normalize import dinovol_normalise
    x = torch.from_numpy(dinovol_normalise(u8).astype(np.float32))[None, None]
    with torch.inference_mode():
        feats = net.backbone.forward_features(x, view_kind="global")
        tok = feats["x_norm_patchtokens"]
        B, G, C = tok.shape; side = round(G ** (1 / 3))
        dec = net.pixel_decoder(tok.transpose(1, 2).reshape(B, C, side, side, side), feats["embed_skips"])
        inst = net.instance_head(dec["memory"], dec["mask_features"])
        sem = dec["sem_logits"][0][0]
        me = inst["layers"][-1]["mask_embed"][0]
        ml = torch.einsum("qc,cdhw->qdhw", me, dec["mask_features"][0])
        owner = ml.argmax(0)
        memb = torch.exp(ml.amax(0) - torch.logsumexp(ml, 0))
    return {"tokens": tok[0].reshape(side, side, side, C).numpy(), "margin": (sem[1] - sem[0]).numpy(), "embed": me.numpy(),
            "owner": owner.numpy().astype(np.uint8), "memb": memb.numpy(), "skips": [s[0].permute(1, 2, 3, 0).numpy() for s in feats["embed_skips"]]}

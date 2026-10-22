"""Reference forward of the scrollprize dinovol model (HF scrollprize/dinovol_8um_ps192_102126) built from the official
dinovol_2 sources (ScrollPrize/villa branch dinovol-refactor), fp32 on the CPU, for tests/test_dinovol.py only (never part of a
pipeline). Data dir UFSM_DINOVOL (default ~/ufsm-data/dinovol): teacher_step_640000.pt, src/dinovol_2 (the official package),
pylib (timm + einops installed with pip --target --no-deps). timm's package __init__ (torchvision) is bypassed and the one
torchvision symbol timm.layers imports is a stand-in, so only torch and numpy are needed.

run_tile(u8 [T,T,T], center, scale, hi, points) mirrors eval/embed_volume.py on one tile: Normalize("fixed") with clip to hi,
backbone.forward_features(view_kind="global") with embed skips, x_norm_patchtokens, then the PointDenseHead (fpn_grid,
sample_grid_points, fpn_norm, readout) at the given voxel-index points (+0.5 = continuous centers)."""
import os, pathlib, sys, types

DATA = pathlib.Path(os.environ.get("UFSM_DINOVOL", "~/ufsm-data/dinovol")).expanduser()
CKPT = DATA / "teacher_step_640000.pt"


def setup():
    lib = DATA / "pylib"
    sys.path.insert(0, str(DATA / "src"))
    sys.path.insert(0, str(lib))
    for n in ["torchvision", "torchvision.ops", "torchvision.ops.misc"]:
        m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    sys.modules["torchvision.ops.misc"].FrozenBatchNorm2d = type("FrozenBatchNorm2d", (), {})
    m = types.ModuleType("timm"); m.__path__ = [str(lib / "timm")]; sys.modules["timm"] = m


def build():
    import torch
    setup()
    from dinovol_2.model.backbone import build_backbone
    from dinovol_2.model.model import PointDenseHead
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    cfg = ck["config"]["model"]; ck = ck["teacher"]
    bb = build_backbone(cfg)
    sd = {k[len("backbone."):]: v for k, v in ck.items() if isinstance(k, str) and k.startswith("backbone.")}
    bb.load_state_dict(sd, strict=True)
    bb.eval()
    hs = {k[len("dense_head."):]: v for k, v in ck.items() if isinstance(k, str) and k.startswith("dense_head.")}
    out_dim, hidden = hs["obj_proj.weight"].shape
    head = PointDenseHead(token_dim=bb.embed_dim, skip_channels=tuple(bb.down_projection.skip_channels), hidden_dim=int(hidden),
                          out_dim=int(out_dim), ink_dim=int(hs["ink_proj.weight"].shape[0]), fpn_dim=int(hs["fpn_in.weight"].shape[0]))
    head.load_state_dict(hs, strict=True)
    head.eval()
    return bb, head


def configure(bb, tile):
    t = (tile,) * 3
    bb.global_crops_size = t; bb.local_crops_size = t
    bb.global_ref_feat_shape = tuple(s // p for s, p in zip(t, bb.patch_size)); bb.local_ref_feat_shape = bb.global_ref_feat_shape


def run_tile(bb, head, u8, center, scale, hi, points=None):
    import numpy as np, torch
    from dinovol_2.ops.point_embeddings import sample_grid_points
    T = u8.shape[0]
    configure(bb, T)
    x = (np.minimum(u8.astype(np.float64), hi) - center) / scale
    x = torch.from_numpy(x.astype(np.float32))[None, None]
    with torch.inference_mode():
        bb.return_embed_skips = True
        out = bb.forward_features(x, masks=None, view_kind="global")
        tok = out["x_norm_patchtokens"][0]
        g = T // 8
        grid = tok.reshape(g, g, g, -1)
        res = {"tokens": grid.numpy(), "skips": [s[0].permute(1, 2, 3, 0).numpy() for s in out["embed_skips"]]}
        fpn = head.fpn_grid(grid[None], out["embed_skips"])
        res["fpn"] = fpn[0].numpy()
        if points is not None:
            p = torch.from_numpy(points.astype(np.float32))[None]
            shp = (T, T, T)
            fpn_pts = head.fpn_norm(sample_grid_points(fpn, p, shp))
            skips = [s.permute(0, 2, 3, 4, 1) for s in out["embed_skips"]]
            r = head.readout(sample_grid_points(grid[None], p, shp), [sample_grid_points(s, p, shp) for s in skips], fpn_pts, dtype=torch.float32)
            res["dense"] = r["dense"][0].numpy(); res["ink"] = r["ink"][0].numpy()
    return res

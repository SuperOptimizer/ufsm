"""Independent PyTorch references of the ink-sdk profile networks (https://github.com/LimeGS/ink-sdk), for the ufsm ink-ext parity tests
only (never part of a pipeline). Written from each architecture's description (layer list, state-dict names and shapes, the profiles'
preprocessing), in plain torch.nn.functional over the checkpoint's own tensors: no upstream / ink-sdk / villa module is imported or
executed. fp32 on the CPU.

Networks:
- koine(sd, tiles): the ink_9um family (model_type vesuvius_unet_3d_stem_2d; Reader V2 / V3, rectina, dense-native, TRACE v3B, the
  scrollprize seeds): robust_mad normalisation per tile (p1 / p99 clip, median, 1.4826 MAD), a 3-D depth-fusion stem (two 3x3x3 convs
  + InstanceNorm + LeakyReLU 0.01, a 1x1x1 attention logit, softmax over depth: the attention-weighted sum ++ the max over depth),
  then a 2-D residual U-Net (BasicBlockD stages, stride-2 first convs with an avg-pool skip, transposed-conv decoder, conv + IN +
  LReLU per stage), a 1x1 head -> the ink logits.
- canonical(sd, tiles): canonical_2um (ResNet3D-152 bottleneck encoder, kenshohara layout, stem 7^3 stride (1, 2, 2) + (1, 3, 3)
  max-pool, shortcut B; the 3-D U-Net decoder: 1x1x1 channel reduce + GroupNorm(min(32, c)) + ReLU, trilinear up, residual
  conv blocks with GroupNorm; a depth-attention collapse (1x1x1 logit, softmax over depth) and a 1x1 logit at 1/4 resolution):
  input u8 / 255 through the checkpoint's BatchNorm3d(1); the logits upsampled as probabilities (bilinear, align_corners False)
  and returned as their logits (clamped to float eps), the T x T map ink-ext writes.
- studio(sd, tiles): Studio DeepResidualUNet (widths 48 96 192 384 640; pre-activation residual blocks GroupNorm(gcd(32, c)) + SiLU +
  3x3x3 conv twice; GroupNorm + SiLU + strided 3x3x3 down convs (first (1, 2, 2), then 2); decoder: trilinear resize, 1x1 up conv,
  concat with the skip, GroupNorm + SiLU + 1x1 fuse, residual blocks; head GroupNorm + SiLU + 1x1 conv; the logits weighted by
  softmax(depth_logits) and summed over the 16 planes): input clamp(u8, 0, 200) / 255; the tile's logits (128 x 128).
- timesformer(sd, tiles): the ink TimeSformers (lucidrains TimeSformer-pytorch 0.4.1 semantics: patch rows (p1 p2 c) of 16 px, a cls
  token, per layer pre-LayerNorm time attention, space attention (dim_head 64; the cls query over all tokens unrotated, patch queries
  over the cls token + their frame column / frame row with rotary q / k: frame RotaryEmbedding, axial 4 x 4 patch embedding,
  rotate_every_two), pre-LN GEGLU feed-forward; LayerNorm + Linear(512, 16) of the cls token = 4 x 4 logits): input clip(u8, 0, 200)
  / 200, frames = the 26 layers in tile order; the 4 x 4 logits upsampled as probabilities (bilinear) and returned as logits.
Each function takes uint8 tiles [B][D][T][T] (numpy) and returns float64 logits [B][T][T]."""
import numpy as np


def load_state(path, prefer=("ema_model", "model", "state_dict")):
    """a checkpoint's tensors as {name: torch.float32}: safetensors directly, .pth / .ckpt through torch.load(weights_only=True) with
    an allowlist of inert training-state classes (no code from the file runs)"""
    import torch
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return {k: v.float() for k, v in load_file(path).items()}
    class _Inert:   # optimizer / scheduler objects pickled into training checkpoints: kept as opaque data
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): pass
    allowed = [(_Inert, n) for n in ("torch.optim.adamw.AdamW", "torch.optim.adam.Adam", "torch.optim.lr_scheduler.CosineAnnealingLR",
                                     "torch.optim.lr_scheduler.LambdaLR", "torch.optim.lr_scheduler.OneCycleLR")]
    with torch.serialization.safe_globals([*allowed, (lambda *a: {}, "collections.defaultdict"), dict]):
        d = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(d, dict) and all(hasattr(v, "shape") for v in d.values()):
        sd = d
    else:
        sd = next(d[k] for k in prefer if k in d and isinstance(d[k], dict) and d[k])
    return {k: v.float() for k, v in sd.items() if hasattr(v, "dtype") and v.dtype.is_floating_point}


def robust_mad(x):
    """per-tile robust normalisation: clip to the (numpy linear) 1st / 99th percentiles, centre on the clipped median, divide by
    1.4826 x the clipped MAD (fallbacks: the pre-clip std, half the percentile span, 1)"""
    x = x.astype(np.float64)
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
    c = np.clip(x, lo, hi)
    med = np.median(c)
    s = 1.4826 * np.median(np.abs(c - med))
    if not s >= 1e-6:
        s = x.std() if x.std() >= 1e-6 else (abs(hi - lo) / 2 if abs(hi - lo) / 2 >= 1e-6 else 1.0)
    return ((c - med) / s).astype(np.float32)


def _inorm(x, sd, pre):
    import torch.nn.functional as F
    return F.instance_norm(x, weight=sd[pre + ".weight"], bias=sd[pre + ".bias"], eps=1e-5)


def koine(sd, tiles):
    import torch, torch.nn.functional as F
    lrelu = lambda t: F.leaky_relu(t, 0.01)
    E, Dd = "network.shared_encoder.", "network.shared_decoder."
    out = []
    with torch.inference_mode():
        for t in tiles:
            x = torch.from_numpy(robust_mad(t))[None, None]                                   # [1][1][D][T][T]
            f = lrelu(_inorm(F.conv3d(x, sd["depth_fusion.features.0.weight"], padding=1), sd, "depth_fusion.features.1"))
            f = lrelu(_inorm(F.conv3d(f, sd["depth_fusion.features.3.weight"], padding=1), sd, "depth_fusion.features.4"))
            a = F.conv3d(f, sd["depth_fusion.attention_logits.weight"], sd["depth_fusion.attention_logits.bias"]).softmax(2)
            x = torch.cat([(f * a).sum(2), f.amax(2)], 1)                                      # [1][2c][T][T]
            x = lrelu(_inorm(F.conv2d(x, sd[E + "stem.convs.0.conv.weight"], sd[E + "stem.convs.0.conv.bias"], padding=1), sd, E + "stem.convs.0.norm"))
            skips = []
            s = 0
            while E + f"stages.{s}.blocks.0.conv1.conv.weight" in sd:
                b = 0
                while (p := E + f"stages.{s}.blocks.{b}.") + "conv1.conv.weight" in sd:
                    st = 2 if s > 0 and b == 0 else 1
                    h = lrelu(_inorm(F.conv2d(x, sd[p + "conv1.conv.weight"], sd[p + "conv1.conv.bias"], stride=st, padding=1), sd, p + "conv1.norm"))
                    h = _inorm(F.conv2d(h, sd[p + "conv2.conv.weight"], sd[p + "conv2.conv.bias"], padding=1), sd, p + "conv2.norm")
                    r = F.avg_pool2d(x, 2) if st == 2 else x
                    if p + "skip.1.conv.weight" in sd:
                        r = _inorm(F.conv2d(r, sd[p + "skip.1.conv.weight"]), sd, p + "skip.1.norm")
                    x = lrelu(h + r)
                    b += 1
                skips.append(x)
                s += 1
            x = skips[-1]
            for k in range(len(skips) - 1):
                x = F.conv_transpose2d(x, sd[Dd + f"transpconvs.{k}.weight"], sd[Dd + f"transpconvs.{k}.bias"], stride=2)
                x = torch.cat([x, skips[-2 - k]], 1)
                p = Dd + f"stages.{k}.convs.0."
                x = lrelu(_inorm(F.conv2d(x, sd[p + "conv.weight"], sd[p + "conv.bias"], padding=1), sd, p + "norm"))
            y = F.conv2d(x, sd["network.task_heads.ink.weight"], sd["network.task_heads.ink.bias"])
            out.append(y[0, 0].double().numpy())
    return np.stack(out)


def read_tif(path):
    """ufsm tiffw's uint8 strip TIFFs (deflate or none) -> [h][w] uint8"""
    import struct, zlib
    b = open(path, "rb").read(); off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; tg = {}
    for i in range(n):
        t, ty, c = struct.unpack_from("<HHI", b, off + 2 + 12 * i); v = off + 2 + 12 * i + 8
        tg[t] = [struct.unpack_from("<H" if ty == 3 else "<I", b, v)[0]] if c == 1 else list(struct.unpack_from(f"<{c}I", b, struct.unpack_from("<I", b, v)[0]))
    w, h = tg[256][0], tg[257][0]; data = b""
    for o, l in zip(tg[273], tg[279]):
        s = b[o:o + l]; data += zlib.decompress(s) if tg[259][0] == 8 else s
    return np.frombuffer(data, np.uint8)[: w * h].reshape(h, w)


def write_tif(path, a):
    """uint8 [h][w] -> one uncompressed strip TIFF (ink-ext --layers input)"""
    import struct
    h, w = a.shape
    tags = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8 + 2 + 12 * 9 + 4),
            (277, 3, 1, 1), (278, 3, 1, h), (279, 4, 1, w * h)]
    with open(path, "wb") as f:
        f.write(b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(tags)))
        for t, ty, n, v in tags:
            f.write(struct.pack("<HHI", t, ty, n) + (struct.pack("<HH", v, 0) if ty == 3 else struct.pack("<I", v)))
        f.write(struct.pack("<I", 0)); f.write(a.astype(np.uint8).tobytes())


def real_tiles(stack_dir, D, T, n, seed=0):
    """n tiles [n][D][T][T] cut from a rendered surface-volume stack (ink-ext --render-only --dump-layers DIR: DIR/NN.tif) at tile
    origins with mostly data; None when the stack is absent"""
    import pathlib
    p = pathlib.Path(stack_dir)
    fs = sorted(p.glob("[0-9][0-9].tif"))
    if len(fs) < D:
        return None
    st = np.stack([read_tif(f) for f in fs[:D]])
    H, W = st.shape[1:]
    if H < T or W < T:
        return None
    rng = np.random.default_rng(seed); out = []
    for _ in range(200):
        y, x = int(rng.integers(0, H - T + 1)), int(rng.integers(0, W - T + 1))
        t = st[:, y:y + T, x:x + T]
        if (t.max(0) > 0).mean() > 0.6:
            out.append(t.copy())
        if len(out) == n:
            break
    return np.stack(out) if out else None


def _bn(x, sd, pre, eps=1e-5):
    import torch.nn.functional as F
    return F.batch_norm(x, sd[pre + ".running_mean"], sd[pre + ".running_var"], sd[pre + ".weight"], sd[pre + ".bias"], False, 0.0, eps)


def _resnet3d(sd, x, pre, blocks):
    import torch.nn.functional as F
    feats = []
    for L, nb in enumerate(blocks):
        for i in range(nb):
            p = f"{pre}layer{L + 1}.{i}."
            s = 2 if L > 0 and i == 0 else 1
            h = F.relu(_bn(F.conv3d(x, sd[p + "conv1.weight"]), sd, p + "bn1"))
            h = F.relu(_bn(F.conv3d(h, sd[p + "conv2.weight"], stride=s, padding=1), sd, p + "bn2"))
            h = _bn(F.conv3d(h, sd[p + "conv3.weight"]), sd, p + "bn3")
            r = _bn(F.conv3d(x, sd[p + "downsample.0.weight"], stride=s), sd, p + "downsample.1") if p + "downsample.0.weight" in sd else x
            x = F.relu(h + r)
        feats.append(x)
    return feats


def canonical(sd, tiles, divisor=255.0):
    import torch, torch.nn.functional as F
    gn = lambda t, p: F.group_norm(t, min(32, t.shape[1]), sd[p + ".weight"], sd[p + ".bias"], 1e-5)
    out = []
    with torch.inference_mode():
        for t in tiles:
            T = t.shape[-1]
            x = torch.from_numpy(t.astype(np.float32) / divisor)[None, None]
            if "normalization.weight" in sd:
                x = _bn(x, sd, "normalization")
            x = F.relu(_bn(F.conv3d(x, sd["backbone.conv1.weight"], stride=(1, 2, 2), padding=3), sd, "backbone.bn1"))
            x = F.max_pool3d(x, (1, 3, 3), (1, 2, 2), (0, 1, 1))
            f = _resnet3d(sd, x, "backbone.", [3, 8, 36, 3])
            r = [F.relu(gn(F.conv3d(f[i], sd[f"decoder.channel_reduce.{i}.0.weight"]), f"decoder.channel_reduce.{i}.1")) for i in range(4)]
            x = r[3]
            for j in range(3):
                sk = r[2 - j]
                x = torch.cat([F.interpolate(x, size=sk.shape[2:], mode="trilinear", align_corners=False), sk], 1)
                p = f"decoder.decoder_blocks.{j}."
                idn = F.conv3d(x, sd[p + "shortcut.weight"]) if p + "shortcut.weight" in sd else x
                h = F.relu(gn(F.conv3d(x, sd[p + "conv1.weight"], padding=1), p + "gn1"))
                x = F.relu(gn(F.conv3d(h, sd[p + "conv2.weight"], padding=1), p + "gn2") + idn)
            a = F.conv3d(x, sd["decoder.depth_collapse.attn_conv.weight"], sd["decoder.depth_collapse.attn_conv.bias"]).softmax(2)
            y = F.conv2d((x * a).sum(2), sd["decoder.logit.weight"], sd["decoder.logit.bias"])
            pr = F.interpolate(y.sigmoid(), size=(T, T), mode="bilinear", align_corners=False)[0, 0].double()
            eps = 1.1920929e-07
            pr = pr.clamp(eps, 1 - eps)
            out.append(torch.log(pr / (1 - pr)).numpy())
    return np.stack(out)


def studio(sd, tiles):
    import math, torch, torch.nn.functional as F
    widths, enc_n, dec_n = [48, 96, 192, 384, 640], [2, 3, 4, 5, 6], [3, 3, 2, 2]
    ns = lambda t, p: F.silu(F.group_norm(t, math.gcd(32, t.shape[1]), sd[p + ".weight"], sd[p + ".bias"], 1e-5))
    def block(x, p):
        h = F.conv3d(ns(x, p + ".branch.0"), sd[p + ".branch.2.weight"], padding=1)
        return x + F.conv3d(ns(h, p + ".branch.3"), sd[p + ".branch.6.weight"], padding=1)
    out = []
    with torch.inference_mode():
        for t in tiles:
            x = torch.from_numpy(np.clip(t.astype(np.float32), 0, 200) / 255)[None, None]
            x = F.conv3d(x, sd["stem.weight"], padding=1); skips = []
            for i in range(5):
                for j in range(enc_n[i]):
                    x = block(x, f"enc.{i}.{j}")
                if i < 4:
                    skips.append(x)
                    x = F.conv3d(ns(x, f"down.{i}.0"), sd[f"down.{i}.2.weight"], stride=(1 if i == 0 else 2, 2, 2), padding=1)
            for k in range(4):
                sk = skips[3 - k]
                x = F.conv3d(F.interpolate(x, size=sk.shape[-3:], mode="trilinear", align_corners=False), sd[f"up.{k}.weight"])
                x = F.conv3d(ns(torch.cat([x, sk], 1), f"fuse.{k}.0"), sd[f"fuse.{k}.2.weight"])
                for j in range(dec_n[k]):
                    x = block(x, f"dec.{k}.{j}")
            h = F.conv3d(ns(x, "head.0"), sd["head.2.weight"], sd["head.2.bias"])
            y = (h * sd["depth_logits"].softmax(0)[None, None, :, None, None]).sum(2)
            out.append(y[0, 0].double().numpy())
    return np.stack(out)


def timesformer(sd, tiles):
    import math, torch, torch.nn.functional as F
    P = next(k[:-len("cls_token")] for k in sd if k.endswith("cls_token"))
    g = lambda k: sd[P + k]
    depth = sum(1 for k in sd if k.startswith(P + "layers.") and k.endswith(".0.fn.to_qkv.weight"))
    H = g("layers.0.0.fn.to_qkv.weight").shape[0] // (3 * 64)
    Fr, NP, C = 26, 16, 512
    # rotary tables (fp32, as the modules compute them from their buffers)
    fr = torch.arange(Fr, dtype=torch.float32)[:, None] * g("frame_rot_emb.inv_freqs")[None, :]
    fr = torch.cat([fr, fr], -1); rot_t = (fr.sin(), fr.cos())                            # [26][64]
    hs = torch.linspace(-1., 1., 4)[:, None] * g("image_rot_emb.scales")[None, :] * math.pi  # [4][16]
    xs = hs[:, None, :].expand(4, 4, 16); ys = hs[None, :, :].expand(4, 4, 16)
    sn = torch.cat([xs.sin(), ys.sin()], -1).reshape(16, 32).repeat_interleave(2, -1)
    cs = torch.cat([xs.cos(), ys.cos()], -1).reshape(16, 32).repeat_interleave(2, -1)
    rot_s = (sn, cs)                                                                       # [16][64]
    def rot2(t):   # rotate_every_two
        t2 = t.reshape(*t.shape[:-1], -1, 2)
        return torch.stack([-t2[..., 1], t2[..., 0]], -1).reshape(t.shape)
    def attention(x, pre, time):
        b, n, _ = x.shape
        q, k, v = F.linear(x, sd[pre + "fn.to_qkv.weight"]).chunk(3, -1)
        sp = lambda t: t.reshape(b, n, H, 64).permute(0, 2, 1, 3).reshape(b * H, n, 64)
        q, k, v = sp(q) * 64 ** -0.5, sp(k), sp(v)
        cls_out = (q[:, :1] @ k.transpose(1, 2)).softmax(-1) @ v                           # [bH][1][64]
        qr, kr, vr = (t[:, 1:].reshape(b * H, Fr, NP, 64) for t in (q, k, v))
        if time:   # (b n) f d
            qr, kr, vr = (t.permute(0, 2, 1, 3).reshape(b * H * NP, Fr, 64) for t in (qr, kr, vr)); sin, cos = rot_t
        else:      # (b f) n d
            qr, kr, vr = (t.reshape(b * H * Fr, NP, 64) for t in (qr, kr, vr)); sin, cos = rot_s
        qr = qr * cos + rot2(qr) * sin; kr = kr * cos + rot2(kr) * sin
        r = qr.shape[0] // (b * H)
        kc = torch.cat([k[:, :1].repeat_interleave(r, 0), kr], 1); vc = torch.cat([v[:, :1].repeat_interleave(r, 0), vr], 1)
        o = (qr @ kc.transpose(1, 2)).softmax(-1) @ vc
        o = o.reshape(b * H, NP, Fr, 64).permute(0, 2, 1, 3) if time else o.reshape(b * H, Fr, NP, 64)
        o = torch.cat([cls_out, o.reshape(b * H, Fr * NP, 64)], 1)
        o = o.reshape(b, H, n, 64).permute(0, 2, 1, 3).reshape(b, n, H * 64)
        return F.linear(o, sd[pre + "fn.to_out.0.weight"], sd[pre + "fn.to_out.0.bias"])
    ln = lambda x, pre: F.layer_norm(x, (C,), sd[pre + ".weight"], sd[pre + ".bias"], 1e-5)
    out = []
    with torch.inference_mode():
        for t in tiles:
            v = torch.from_numpy(np.clip(t.astype(np.float32), 0, 200) / 200)                 # [26][64][64] = (f h w)
            v = v.reshape(Fr, 4, 16, 4, 16).permute(0, 1, 3, 2, 4).reshape(1, Fr * NP, 256)  # b (f h w) (p1 p2 c)
            x = torch.cat([g("cls_token")[None], F.linear(v, g("to_patch_embedding.weight"), g("to_patch_embedding.bias"))], 1)
            for L in range(depth):
                x = attention(ln(x, P + f"layers.{L}.0.norm"), P + f"layers.{L}.0.", True) + x
                x = attention(ln(x, P + f"layers.{L}.1.norm"), P + f"layers.{L}.1.", False) + x
                h = F.linear(ln(x, P + f"layers.{L}.2.norm"), g(f"layers.{L}.2.fn.net.0.weight"), g(f"layers.{L}.2.fn.net.0.bias"))
                a, gate = h.chunk(2, -1)
                x = F.linear(a * F.gelu(gate), g(f"layers.{L}.2.fn.net.3.weight"), g(f"layers.{L}.2.fn.net.3.bias")) + x
            y = F.linear(F.layer_norm(x[:, 0], (C,), g("to_out.0.weight"), g("to_out.0.bias"), 1e-5), g("to_out.1.weight"), g("to_out.1.bias"))
            pr = F.interpolate(y.reshape(1, 1, 4, 4).sigmoid(), size=(64, 64), mode="bilinear", align_corners=False)[0, 0].double()
            eps = 1.1920929e-07
            pr = pr.clamp(eps, 1 - eps)
            out.append(torch.log(pr / (1 - pr)).numpy())
    return np.stack(out)

"""Reference forward of the published 3D ink network scrollprize/ink_paris4_3d_4.8um_gambiarra (HF, MIT), fp32 on the CPU, for
tests/test_ink3d_gambiarra.py only (never part of a pipeline).

The net is vesuvius NetworkFromConfig "unet" (villa branch ink-paris4-3d-gambiarra, vesuvius/models/build/{encoder,decoder,
resblocks,simple_conv_blocks}.py) rebuilt here with the same module names, so model.safetensors loads strictly:
encoder stages of BasicBlockD (features 32 64 128 256 320, blocks 1 2 2 2 2, the first block of a stage strided 1 2 2 2 2;
conv3 + InstanceNorm3d(affine, eps 1e-5) + LeakyReLU(0.01), conv3 + IN, skip = [AvgPool3d(2)] + [1x1 conv (no bias) + IN] when
the stride or width changes, add, LeakyReLU), decoder per stage ConvTranspose3d(k 2, s 2) -> cat(up, skip) -> one conv3 + IN +
LeakyReLU, head task_heads.ink = 1x1 conv to one logit.

Input conventions (paris4_3d_gambiarra/volume.py): crop = level-0 [2a, 2a + 128)^3, parent = [2a - 64, 2a + 192)^3, p1 / p99 by
np.percentile (linear) of the parent, clip / subtract / divide in float64 rounded to float32 at each step, avg_pool3d(2) -> 64^3."""
import os, pathlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS = pathlib.Path(os.environ.get("UFSM_GAMBIARRA", "~/ufsm-data/inkft/teachers/gambiarra")).expanduser() / "model.safetensors"
FEAT = [32, 64, 128, 256, 320]
NBLK = [1, 2, 2, 2, 2]


class CNR(nn.Module):
    """ConvDropoutNormReLU: conv -> InstanceNorm -> optional LeakyReLU"""
    def __init__(self, ci, co, k, s, bias, act):
        super().__init__()
        self.conv = nn.Conv3d(ci, co, k, s, (k - 1) // 2, bias=bias)
        self.norm = nn.InstanceNorm3d(co, eps=1e-5, affine=True)
        self.act = act

    def forward(self, x):
        x = self.norm(self.conv(x))
        return F.leaky_relu(x, 0.01) if self.act else x


class BlockD(nn.Module):
    def __init__(self, ci, co, s):
        super().__init__()
        self.conv1 = CNR(ci, co, 3, s, True, True)
        self.conv2 = CNR(co, co, 3, 1, True, False)
        ops = []
        if s != 1:
            ops.append(nn.AvgPool3d(s, s))
        if ci != co:
            ops.append(CNR(ci, co, 1, 1, False, False))
        self.skip = nn.Sequential(*ops) if ops else None

    def forward(self, x):
        r = self.skip(x) if self.skip is not None else x
        return F.leaky_relu(self.conv2(self.conv1(x)) + r, 0.01)


class Stage(nn.Module):
    def __init__(self, ci, co, n, s):
        super().__init__()
        self.blocks = nn.Sequential(*[BlockD(ci if i == 0 else co, co, s if i == 0 else 1) for i in range(n)])

    def forward(self, x):
        return self.blocks(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        st, ci = [], 1
        for s in range(5):
            st.append(Stage(ci, FEAT[s], NBLK[s], 1 if s == 0 else 2))
            ci = FEAT[s]
        self.stages = nn.Sequential(*st)


class DecStage(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.convs = nn.Sequential(CNR(ci, co, 3, 1, True, True))

    def forward(self, x):
        return self.convs(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transpconvs = nn.ModuleList([nn.ConvTranspose3d(FEAT[-s] if s > 1 else FEAT[-1], FEAT[-(s + 1)], 2, 2) for s in range(1, 5)])
        self.stages = nn.ModuleList([DecStage(2 * FEAT[-(s + 1)], FEAT[-(s + 1)]) for s in range(1, 5)])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = Encoder()
        self.shared_decoder = Decoder()
        self.task_heads = nn.ModuleDict({"ink": nn.Conv3d(32, 1, 1)})

    def forward(self, x):
        skips = []
        for s in self.shared_encoder.stages:
            x = s(x)
            skips.append(x)
        x = skips[-1]
        for i in range(4):
            x = self.shared_decoder.transpconvs[i](x)
            x = torch.cat((x, skips[-(i + 2)]), 1)
            x = self.shared_decoder.stages[i](x)
        return self.task_heads["ink"](x)


def load_safetensors(path):
    """{name: torch tensor} of a safetensors file (8-byte header length, JSON header, raw little-endian data)"""
    import json, struct
    raw = pathlib.Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    dt = {"F32": np.float32, "F16": np.float16, "BF16": None, "F64": np.float64}
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = torch.from_numpy(np.frombuffer(raw[8 + n + a:8 + n + b], dt[v["dtype"]]).reshape(v["shape"]).astype(np.float32).copy())
    return out


def build():
    net = Net()
    net.load_state_dict(load_safetensors(WEIGHTS), strict=True)
    assert sum(p.numel() for p in net.parameters()) == 24500769
    return net.eval()


def percentiles(parent):
    """np.percentile(parent as float32, [1, 99]) (linear), as volume.cpu_check"""
    return np.percentile(parent.astype(np.float32), [1, 99])


def normalize(raw, lo, hi):
    """volume.normalize_reference"""
    if hi - lo < 1e-6:
        return np.zeros(raw.shape, np.float32)
    x = np.clip(raw.astype(np.float64), lo, hi).astype(np.float32)
    x = (x.astype(np.float64) - lo).astype(np.float32)
    return (x.astype(np.float64) / (hi - lo)).astype(np.float32)


def run_parents(net, parents):
    """parents uint8 [N, 256, 256, 256] -> (lo [N], hi [N], logits float32 [N, 64, 64, 64])"""
    los, his, imgs = [], [], []
    for p in parents:
        lo, hi = percentiles(p)
        los.append(lo); his.append(hi)
        imgs.append(normalize(p[64:192, 64:192, 64:192], lo, hi))
    x = torch.from_numpy(np.stack(imgs)[:, None])
    with torch.no_grad():
        y = net(F.avg_pool3d(x, 2))
    return np.array(los), np.array(his), y[:, 0].numpy()

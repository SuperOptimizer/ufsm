#!/usr/bin/env python3
"""The production unwrap U-Net in PyTorch, for a speed comparison with ufsm's CUDA path (tests/prof_infer, ~/ufsm-data/pi_prod.sh).

Same layers as src/unet.c with the desk recipe: 6 levels, encoder widths 32,64,96,128,160,192 (two 3^3 convs + GroupNorm(8) + SiLU
per level, from 4 input channels), stride-2 3^3 down convs with GroupNorm + SiLU (--down-norm 1), decoder widths 160,128,96,32,32
(--dec-widths 0,32: trilinear x2 upsample (half-pixel), concat with the skip, two 3^3 convs), a 1^3 head to 11 logits.

Timed as prof_infer times ufsm: a training step = forward + backward from a fixed logit gradient (no loss, no optimizer), inference =
forward. The PyTorch side gets every stock speed-up: bf16 autocast, channels_last_3d, cudnn.benchmark, TF32, torch.compile
(inductor; GN + SiLU fused into one kernel), optional per-block activation checkpointing so larger windows fit, CUDA graphs for
inference. Usage: torch_compare.py --P 192 [--mode train|infer] [--compile none|default|max] [--ckpt 0|1] [--iters 10]
"""
import argparse, time, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

W = [32, 64, 96, 128, 160, 192]
D = [32, 32, 96, 128, 160]   # decoder output width per level (dec-widths 0,32)
G = 8

def gn(c): return nn.GroupNorm(min(G, c), c)

class Block(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.c1, s.n1, s.c2, s.n2 = nn.Conv3d(cin, cout, 3, padding=1), gn(cout), nn.Conv3d(cout, cout, 3, padding=1), gn(cout)
    def forward(s, x): return F.silu(s.n2(s.c2(F.silu(s.n1(s.c1(x))))))

class UNet(nn.Module):
    def __init__(s, cin=4, cout=11, ckpt=False):
        super().__init__()
        L = len(W); s.L, s.ckpt = L, ckpt
        s.enc = nn.ModuleList([Block(cin if i == 0 else W[i - 1], W[i]) for i in range(L)])
        s.down = nn.ModuleList([nn.Conv3d(W[i], W[i], 3, stride=2, padding=1) for i in range(L - 1)])
        s.dn = nn.ModuleList([gn(W[i]) for i in range(L - 1)])
        up = lambda i: W[i] if i == L - 1 else D[i]
        s.dec = nn.ModuleList([Block(W[i] + up(i + 1), D[i]) for i in range(L - 1)])
        s.head = nn.Conv3d(W[0], cout, 1)
    def run(s, m, x): return checkpoint(m, x, use_reentrant=False) if s.ckpt and s.training else m(x)
    def forward(s, x):
        skips = []
        for i in range(s.L):
            x = s.run(s.enc[i], x)
            if i < s.L - 1: skips.append(x); x = F.silu(s.dn[i](s.down[i](x)))
        for i in range(s.L - 2, -1, -1):
            x = torch.cat([skips[i], F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)], 1)
            x = s.run(s.dec[i], x)
        return s.head(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=192); ap.add_argument("--mode", default="train"); ap.add_argument("--compile", default="none")
    ap.add_argument("--ckpt", type=int, default=0); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--graphs", type=int, default=1)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"; P = a.P; train = a.mode == "train"
    m = UNet(ckpt=bool(a.ckpt)).to(dev).to(memory_format=torch.channels_last_3d)
    x = torch.randn(1, 4, P, P, P, device=dev).to(memory_format=torch.channels_last_3d)
    if not train:   # inference: bf16 weights and input, no autocast bookkeeping
        m = m.to(torch.bfloat16).eval(); x = x.to(torch.bfloat16)
    else: m.train()
    f = m
    if a.compile != "none":
        f = torch.compile(m, mode={"default": None, "max": "max-autotune-no-cudagraphs" if train else "max-autotune"}[a.compile])
    gl = torch.randn(1, 11, P, P, P, device=dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last_3d) * 1e-3
    def step():
        if train:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = f(x)
            y.backward(gl)
            for p in m.parameters(): p.grad = None
        else:
            with torch.inference_mode():
                return f(x)
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        for _ in range(a.warmup): step()
        torch.cuda.synchronize(); tw = time.time() - t0
        ts = []
        for _ in range(a.iters):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); step(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    except torch.OutOfMemoryError:
        print(f"torch P {P} {a.mode} compile={a.compile} ckpt={a.ckpt}: OUT OF MEMORY ({torch.cuda.max_memory_allocated() / 1e9:.1f} GB at the failure)"); return
    ts.sort(); med = ts[len(ts) // 2]
    print(f"torch P {P} {a.mode} compile={a.compile} ckpt={a.ckpt}: median {med:.1f} ms, min {ts[0]:.1f} ms, {P ** 3 / med / 1e3:.1f} Mvox/s, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (warmup {tw:.0f} s)")

if __name__ == "__main__": main()

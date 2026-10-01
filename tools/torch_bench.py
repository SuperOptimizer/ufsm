import torch, torch.nn as nn, torch.nn.functional as F, time, sys
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device("cuda:0")
def block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU())
class UNet(nn.Module):
    def __init__(s, w=(16, 32, 64, 80), cin=4, cout=2):
        super().__init__()
        s.enc = nn.ModuleList([block(cin if i == 0 else w[i-1], w[i]) for i in range(len(w))])
        s.down = nn.ModuleList([nn.Conv3d(c, c, 3, stride=2, padding=1) for c in w[:-1]])
        s.dec = nn.ModuleList([block(w[i] + w[i+1], w[i]) for i in range(len(w)-1)])
        s.head = nn.Conv3d(w[0], cout, 1)
    def forward(s, x):
        skips = []
        for i, e in enumerate(s.enc):
            x = e(x)
            if i < len(s.down): skips.append(x); x = s.down[i](x)
        for i in range(len(s.dec)-1, -1, -1):
            x = F.interpolate(x, size=skips[i].shape[2:], mode="trilinear", align_corners=False)
            x = s.dec[i](torch.cat([x, skips[i]], 1))
        return s.head(x)
P = int(sys.argv[1]) if len(sys.argv) > 1 else 96; B = int(sys.argv[2]) if len(sys.argv) > 2 else 2
m = UNet().to(dev); print("params", sum(p.numel() for p in m.parameters()))
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01)
x = torch.randn(B, 4, P, P, P, device=dev); t = torch.rand(B, 2, P, P, P, device=dev)
def step(amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = m(x); loss = F.binary_cross_entropy_with_logits(y.float(), t)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
for amp in (True, False):
    for _ in range(5): step(amp)
    torch.cuda.synchronize(); t0 = time.time(); n = 20
    for _ in range(n): step(amp)
    torch.cuda.synchronize(); dt = (time.time() - t0) / n
    print(f"torch {'bf16 autocast' if amp else 'fp32/tf32'}: {dt*1000:.1f} ms/step, {B/dt:.1f} samples/s, peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()
# channels_last_3d variant
m = m.to(memory_format=torch.channels_last_3d); x = x.to(memory_format=torch.channels_last_3d)
for _ in range(5): step(True)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20): step(True)
torch.cuda.synchronize(); dt = (time.time() - t0) / 20
print(f"torch bf16 channels_last_3d: {dt*1000:.1f} ms/step, {B/dt:.1f} samples/s")

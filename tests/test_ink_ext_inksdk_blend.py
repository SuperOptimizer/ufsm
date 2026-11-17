#!/usr/bin/env python3
"""ufsm ink-ext tiling + blending of the ink-sdk lanes on a synthetic stack (--layers) against a numpy re-implementation of each
profile's sliding window (from the ink-sdk adapters' descriptions), with ink-ext --raw-in as the per-tile oracle (the network itself is
checked by the parity tests):
- canon2 (canonical2um.py): canonical grid() origins (stride 128, last at L - 256), tiles kept when >= 60 % of their columns hold data,
  window hanning(258)[1:-1] outer / max, prob x window summed over window sums, 0 where the sum is 0 or the column has no data.
- studio (hf/inference.py --stride 64 --blend hann): 128 px windows at (y - 16, x - 16) reflect-padded past the map, starts
  range(0, L, 64) + [max(L - 96, 0)], the central 96 px x a pixel-centred hann(96) summed over the weight sums, 0 where all 16
  planes are 0. A 16 x 230 x 300 stack with an empty corner (ink-ext skips windows whose central 96 px hold no data: exact).
- tsf (timesformer.py): grid() origins of 64 px tiles at stride 16, tiles with any data column, window np.hanning(64) outer (0 on the
  tile edges), 0 where the window sum is 0 or the column has no data. A 26 x 100 x 150 stack with an empty corner.
The canon2 stack (62 x 400 x 520) has an empty band and an empty corner, so tiles are dropped by the 60 % rule and columns zeroed.
|out - round(255 ref)| <= 1 (uint8)."""
import os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D0 = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
WC = D0 / "weights/canonical_2um@r152-3ddec-v2-l5-epoch13/r152_3ddec_v2_l5_epoch13.ckpt"
WS = D0 / "weights/studio@softloss-20261023/studio-softloss-20261023-ema.safetensors"
WT = D0 / "weights/timesformer@gp-scroll1/model.safetensors"
P.need([WC, WS, WT])
import numpy as np
import inksdk_ref as R


def grid(L, T, s):   # canonical2um grid(): range(0, L - T + 1, s) + [L - T]
    return sorted(set(range(0, L - T + 1, s)) | {L - T})


def run_layers(model, w, stack, d):
    (d / "layers").mkdir()
    for i in range(stack.shape[0]):
        R.write_tif(d / "layers" / f"{i:02d}.tif", stack[i])
    r = subprocess.run([P.UFSM, "ink-ext", "--model", model, "--weights", w, "--layers", d / "layers", "--out", d / "o", "--name", "t", "--batch", "2"],
                       capture_output=True, text=True, timeout=1200)
    sys.stderr.write(r.stderr[-2000:])
    assert r.returncode == 0, f"ink-ext --layers {model} failed"
    return R.read_tif(d / "o" / f"t_{model}.tif").astype(np.int32)


def check(name, out, ref):
    err = np.abs(out - np.rint(255 * ref).astype(np.int32))
    print(f"{name}: {out.shape}, max |out - ref blend| {err.max()} / 255, mean {err.mean():.4f}, zero px {int((out == 0).sum())}")
    assert out.shape == ref.shape and err.max() <= 1, f"{name}: blend differs from the profile's sliding window"


rng = np.random.default_rng(11)
# ---- canon2 ----
Dd, H, Wd, T = 62, 400, 520, 256
stack = np.clip(100 + 30 * rng.standard_normal((Dd, H, Wd)), 1, 255).astype(np.uint8)
stack[:, :, 430:] = 0          # an empty band (tiles at x = 264 keep 65 % data columns: kept, their right part zeroed)
stack[:, 250:, :200] = 0       # an empty corner: the x = 0 tiles at y = 128 / 144 hold 59 % / 54 % data columns: dropped
valid = (stack > 0).any(0)
tiles, org = [], []
for y in grid(H, T, 128):
    for x in grid(Wd, T, 128):
        if valid[y:y + T, x:x + T].mean() >= 0.6:
            tiles.append(stack[:, y:y + T, x:x + T]); org.append((y, x))
logit = P.run_c("canon2", WC, np.stack(tiles), batch=2)
w1 = np.hanning(T + 2)[1:-1]; win = np.outer(w1, w1); win /= win.max()
num = np.zeros((H, Wd)); den = np.zeros((H, Wd))
for (y, x), l in zip(org, logit):
    num[y:y + T, x:x + T] += win / (1 + np.exp(-l.astype(np.float64))); den[y:y + T, x:x + T] += win
ref = np.where((den > 0) & valid, num / np.maximum(den, 1e-300), 0)
print(f"canon2: {len(org)} of {len(grid(H, T, 128)) * len(grid(Wd, T, 128))} tiles kept")
with tempfile.TemporaryDirectory(prefix="ufsm-inksdk-blend-") as d:
    check("canon2", run_layers("canon2", WC, stack, pathlib.Path(d)), ref)

# ---- studio ----
Dd, H, Wd = 16, 230, 300
stack = np.clip(100 + 30 * rng.standard_normal((Dd, H, Wd)), 1, 255).astype(np.uint8)
stack[:, 150:, 190:] = 0
starts = lambda L: sorted(set(range(0, L, 64)) | {max(L - 96, 0)})
def crop(a, y, x):   # inference.py crop(): reflect padding past the map
    h, w = a.shape[-2:]
    t = a[:, max(0, y):min(h, y + 128), max(0, x):min(w, x + 128)]
    pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    return np.pad(t, pad, mode="reflect") if any(a or b for a, b in pad) else t
org = [(y, x) for y in starts(H) for x in starts(Wd)]
logit = P.run_c("studio", WS, np.stack([crop(stack, y - 16, x - 16) for y, x in org]), batch=4)
w1 = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96); win = np.outer(w1, w1)
num = np.zeros((H, Wd)); den = np.zeros((H, Wd))
for (y, x), l in zip(org, logit):
    p = 1 / (1 + np.exp(-l[16:112, 16:112].astype(np.float64))); hh, ww = min(96, H - y), min(96, Wd - x)
    num[y:y + hh, x:x + ww] += p[:hh, :ww] * win[:hh, :ww]; den[y:y + hh, x:x + ww] += win[:hh, :ww]
ref = np.where(den > 0, num / np.maximum(den, 1e-300), 0); ref[(stack == 0).all(0)] = 0
with tempfile.TemporaryDirectory(prefix="ufsm-inksdk-blend-") as d:
    check("studio", run_layers("studio", WS, stack, pathlib.Path(d)), ref)

# ---- tsf ----
Dd, H, Wd, T = 26, 100, 150, 64
stack = np.clip(100 + 30 * rng.standard_normal((Dd, H, Wd)), 1, 255).astype(np.uint8)
stack[:, 60:, 90:] = 0
valid = (stack > 0).any(0)
org = [(y, x) for y in grid(H, T, 16) for x in grid(Wd, T, 16) if valid[y:y + T, x:x + T].any()]
logit = P.run_c("tsf", WT, np.stack([stack[:, y:y + T, x:x + T] for y, x in org]), batch=8)
w1 = np.hanning(T); win = np.outer(w1, w1)
num = np.zeros((H, Wd)); den = np.zeros((H, Wd))
for (y, x), l in zip(org, logit):
    num[y:y + T, x:x + T] += win / (1 + np.exp(-l.astype(np.float64))); den[y:y + T, x:x + T] += win
ref = np.where((den > 0) & valid, num / np.maximum(den, 1e-300), 0)
with tempfile.TemporaryDirectory(prefix="ufsm-inksdk-blend-") as d:
    check("tsf", run_layers("tsf", WT, stack, pathlib.Path(d)), ref)
print("test_ink_ext_inksdk_blend: OK")

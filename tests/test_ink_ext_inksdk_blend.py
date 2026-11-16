#!/usr/bin/env python3
"""ufsm ink-ext tiling + blending of the ink-sdk lanes on a synthetic stack (--layers) against a numpy re-implementation of each
profile's sliding window (from the ink-sdk adapters' descriptions), with ink-ext --raw-in as the per-tile oracle (the network itself is
checked by the parity tests):
- canon2 (canonical2um.py): canonical grid() origins (stride 128, last at L - 256), tiles kept when >= 60 % of their columns hold data,
  window hanning(258)[1:-1] outer / max, prob x window summed over window sums, 0 where the sum is 0 or the column has no data.
The stack (62 x 400 x 520) has an empty band and an empty corner, so tiles are dropped by the 60 % rule and columns zeroed.
|out - round(255 ref)| <= 1 (uint8)."""
import os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D0 = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
WC = D0 / "weights/canonical_2um@r152-3ddec-v2-l5-epoch13/r152_3ddec_v2_l5_epoch13.ckpt"
P.need([WC])
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
print("test_ink_ext_inksdk_blend: OK")

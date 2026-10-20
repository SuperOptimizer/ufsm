#!/usr/bin/env python3
"""ufsm ink-ext --model v8in against the official ink8um InkDetector (YoussefMoNader/ink-8um-v8in) in fp32 on the CPU:
tiles (24 x 64 x 64 uint8) clipped to [0, 200] / 255 as inference.py feeds them; logits on the 64 x 64 tile grid."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

W = P.DATA / "weights/v8in.safetensors"
SRC = P.DATA / "src/v8in"
P.need([W, SRC / "ink8um/modeling.py"])
import numpy as np, torch
from safetensors_free import load_safetensors  # noqa: E402  (tests/safetensors_free.py: a numpy reader, no safetensors package needed)

P.stub_hf_hub()
InkDetector = P.load_module("ink8um_modeling", SRC / "ink8um/modeling.py").InkDetector   # the module alone (the package imports cv2)

model = InkDetector().eval()
model.load_state_dict({k: torch.from_numpy(v) for k, v in load_safetensors(W).items()}, strict=True)

tiles = P.random_tiles(3, 24, 64, seed=1)
real = P.real_tiles(3, 24, 64, 1, [(10000, 4000, 4000), (11000, 3600, 4600), (12000, 4400, 3800)])
if real is not None:
    tiles = np.concatenate([tiles, real])
else:
    print("no local CT: random tiles only")

with torch.inference_mode():
    ref = np.concatenate([model(torch.from_numpy(np.clip(t, 0, 200).astype(np.float32)[None] / 255.0)).numpy()[:, 0] for t in tiles])
got = P.run_c("v8in", W, tiles)
P.compare("v8in random", ref[:3], got[:3], atol=0.08, mean_tol=0.02)
if real is not None:
    P.compare("v8in real CT", ref[3:], got[3:], atol=0.08, mean_tol=0.02)
print("test_ink_ext_v8in: OK")

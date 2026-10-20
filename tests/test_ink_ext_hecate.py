#!/usr/bin/env python3
"""ufsm ink-ext --model hecate24 / hecate96 against the official hecate.py (scrollprize/hecate) in fp32 on the CPU: uint8 tiles
(64 x 256 x 256 at 2.4 um, 16 x 64 x 64 at 9.6 um) / divisor with the column support mask hecate.predict passes (raw.any over
depth); the 2D logits (logit of the upsampled probability). One tile has empty columns (the masked softmax)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

SRC = P.DATA / "src/hecate/hecate.py"
CASES = [("hecate24", P.DATA / "weights/hecate_2.4um.pth", 64, 256, 1, [(9000, 4000, 4000), (11500, 3800, 4800)]),
         ("hecate96", P.DATA / "weights/hecate_9.6um.pth", 16, 64, 2, [(5000, 2000, 2000), (5600, 1800, 2300), (6000, 2200, 1900)])]
P.need([SRC] + [c[1] for c in CASES])
import numpy as np, torch
H = P.load_module("hecate_official", SRC)

for name, w, D, T, level, origins in CASES:
    model = H.load_model(str(w), "cpu")
    tiles = P.random_tiles(3, D, T, seed=7)
    tiles[2, :, :, : int(T * 0.4)] = 0   # empty columns: masked out of the depth softmax
    real = P.real_tiles(len(origins), D, T, level, origins)
    if real is not None:
        tiles = np.concatenate([tiles, real])
    else:
        print(f"{name}: no local CT: random tiles only")
    refs = []
    with torch.inference_mode():
        for t in tiles:
            raw = torch.from_numpy(t[None].copy())
            support = raw.any(1)
            image = torch.from_numpy(t[None, None].astype(np.float32) / model.divisor)
            valid = support[:, None, None].expand(-1, 1, D, -1, -1)
            refs.append(model(image, valid).numpy()[:, 0])
    ref = np.concatenate(refs)
    got = P.run_c(name, w, tiles, batch=2)
    P.compare(f"{name} random", ref[:3], got[:3], atol=0.1, mean_tol=0.02)
    if real is not None:
        P.compare(f"{name} real CT", ref[3:], got[3:], atol=0.1, mean_tol=0.02)
    del model
print("test_ink_ext_hecate: OK")

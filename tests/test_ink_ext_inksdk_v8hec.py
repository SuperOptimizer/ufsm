#!/usr/bin/env python3
"""ufsm ink-ext --model v8in (ink-sdk profiles v8in@ink-8um-v8in, v8in@pherc0306b-ahn-s0 / s1 / s2) and hecate24 / hecate96 (hecate@2.4um,
hecate@9.6um) against tests/inksdk_ref.py v8in / hecate (independent torch forwards written from the architectures, fp32 CPU; no
upstream code), on random tiles (one with empty columns) and real surface-volume tiles rendered by ink-ext --render-only --dump-layers
($UFSM_INKSDK/parity/k791_d24, k2399_d64, k9596_d16; PARTIAL without them). v8in: the 64 x 64 logits; Hecate: the T x T logit of its
upsampled probability. ~1-2 CPU minutes per Hecate 2.4 tile."""
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
W = D / "weights"
V8 = [("v8in", W / "v8in@ink-8um-v8in/model.safetensors")] + [(f"ahn-s{s}", W / f"v8in@pherc0306b-ahn-s{s}/model.safetensors") for s in range(3)]
HE = [("hecate24", W / "hecate@2.4um/hecate_2.4um.pth", 64, 256, 2.4, "k2399_d64", 1), ("hecate96", W / "hecate@9.6um/hecate_9.6um.pth", 16, 64, 9.6, "k9596_d16", 3)]
P.need([w for _, w in V8] + [h[1] for h in HE])
import numpy as np
import inksdk_ref as R

tiles = P.random_tiles(2, 24, 64, seed=31)
real = R.real_tiles(D / "parity/k791_d24", 24, 64, 2, seed=8)
if real is None:
    P.partial("no rendered stack k791_d24: v8in on random tiles only")
allt = tiles if real is None else np.concatenate([tiles, real])
for tag, w in V8:
    ref = R.v8in(R.load_state(w), allt)
    got = P.run_c("v8in", w, allt, batch=2)
    P.compare(f"v8in {tag} random", ref[:2], got[:2], atol=0.08, mean_tol=0.02)
    if real is not None:
        P.compare(f"v8in {tag} real surface volume", ref[2:], got[2:], atol=0.08, mean_tol=0.02)
for name, w, Dd, T, um, stk, nreal in HE:
    tiles = P.random_tiles(2, Dd, T, seed=7)[1:]
    tiles[0, :, :, : int(T * 0.4)] = 0   # empty columns: masked out of the depth softmax
    real = R.real_tiles(D / "parity" / stk, Dd, T, nreal, seed=9)
    if real is None:
        P.partial(f"no rendered stack {stk}: {name} on random tiles only")
    allt = tiles if real is None else np.concatenate([tiles, real])
    ref = R.hecate(R.load_state(w), allt, um)
    got = P.run_c(name, w, allt, batch=2)
    P.compare(f"{name} random", ref[:1], got[:1], atol=0.1, mean_tol=0.02)
    if real is not None:
        P.compare(f"{name} real surface volume", ref[1:], got[1:], atol=0.1, mean_tol=0.02)
print("test_ink_ext_inksdk_v8hec: OK")

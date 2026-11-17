#!/usr/bin/env python3
"""ufsm ink-ext --model reader2 | dense9 | rectina | trace3b (the ink-sdk profiles' ink_9um-family checkpoints) and reader3 with Argus V2 against tests/inksdk_ref.py
koine (an independent torch re-implementation of the vesuvius_unet_3d_stem_2d forward with robust_mad normalisation, fp32 CPU) on
uint8 tiles: random patterns, a half-empty edge tile and real surface-volume tiles (an eval crop rendered by ink-ext --render-only
--dump-layers at 9.362 um, 17 layers, in $UFSM_INKSDK/parity/k9362_d17; PARTIAL without it).

Weights: $UFSM_INKSDK (default /vesuvius/ufsm/inksdk)/weights/<profile id>/ (scripts there: fetch.sh, pinned revisions + sha256)."""
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
W = D / "weights"
CASES = [("reader2", W / "ink_9um@reader-v2-step040000/reader-v2-step040000.pth", 128),
         ("dense9", W / "ink_9um@dense-native-step016000/dense_native_016000.safetensors", 128),
         ("trace3b", W / "ink_9um@trace-v3b-step5000/v3B_best.pth", 128),
         ("rectina", W / "ink_9um@rectina-m4/ckpt_020000.pth", 384),
         ("reader3", W / "argus-v2/argus-v2.pth", 384)]   # Argus V2 (ruiconte, a Reader V3 fine-tune, same layout): runs as --model reader3
P.need([c[1] for c in CASES])
import numpy as np
import inksdk_ref as R

for name, w, T in CASES:
    tiles = P.random_tiles(3, 17, T, seed=21)
    tiles[2, :, int(T * 0.55):, :] = 0   # a tile edge beyond the image (zero padded)
    real = R.real_tiles(D / "parity/k9362_d17", 17, T, 3 if T == 128 else 1, seed=5)
    if real is None:
        P.partial(f"no rendered stack in {D}/parity/k9362_d17: {name} on random tiles only")
    allt = tiles if real is None else np.concatenate([tiles, real])
    ref = R.koine(R.load_state(w), allt)
    got = P.run_c(name, w, allt, batch=2)
    P.compare(f"{name} random", ref[:3], got[:3], atol=0.05, mean_tol=0.01)
    if real is not None:
        P.compare(f"{name} real surface volume", ref[3:], got[3:], atol=0.05, mean_tol=0.01)
print("test_ink_ext_inksdk_koine: OK")

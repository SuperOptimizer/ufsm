#!/usr/bin/env python3
"""ufsm ink-ext --model studio (ink-sdk profile studio@softloss-20261023) against tests/inksdk_ref.py studio (an independent torch forward
of the DeepResidualUNet, fp32 CPU) on uint8 tiles 16 x 128 x 128 (clamp(0, 200) / 255): random patterns, an edge tile and real 9.362 um
surface-volume tiles ($UFSM_INKSDK/parity/k9362_d16, an eval crop rendered by ink-ext --render-only --dump-layers; PARTIAL without it).
The compared map is the tile's logits (the depth-softmax sum)."""
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
WS = D / "weights/studio@softloss-20261023/studio-softloss-20261023-ema.safetensors"
P.need([WS])
import numpy as np
import inksdk_ref as R

tiles = P.random_tiles(3, 16, 128, seed=17)
tiles[2, :, :, 80:] = 0
real = R.real_tiles(D / "parity/k9362_d16", 16, 128, 3, seed=4)
if real is None:
    P.partial(f"no rendered stack in {D}/parity/k9362_d16: random tiles only")
allt = tiles if real is None else np.concatenate([tiles, real])
ref = R.studio(R.load_state(WS), allt)
got = P.run_c("studio", WS, allt, batch=3)
P.compare("studio random", ref[:3], got[:3], atol=0.05, mean_tol=0.01)
if real is not None:
    P.compare("studio real surface volume", ref[3:], got[3:], atol=0.05, mean_tol=0.01)
print("test_ink_ext_inksdk_studio: OK")

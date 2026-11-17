#!/usr/bin/env python3
"""ufsm ink-ext --model canon2 (ink-sdk profile canonical_2um@r152-3ddec-v2-l5-epoch13) against tests/inksdk_ref.py canonical (an
independent torch forward of the ResNet3D-152 + 3-D decoder, fp32 CPU) on uint8 tiles 62 x 256 x 256 (u8 / 255 through the checkpoint's
BatchNorm3d(1)): a random tile with empty columns and a real 2.399 um surface-volume tile ($UFSM_INKSDK/parity/k2399_d62, an eval crop
rendered by ink-ext --render-only --dump-layers; PARTIAL without it). The map compared is the T x T logit of the bilinearly upsampled
probability (ink-ext --raw-out). ~2 CPU minutes per tile."""
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
WT = D / "weights/canonical_2um@r152-3ddec-v2-l5-epoch13/r152_3ddec_v2_l5_epoch13.ckpt"
P.need([WT])
import numpy as np
import inksdk_ref as R

tiles = P.random_tiles(2, 62, 256, seed=3)[1:]   # the smooth field
tiles[0, :, :, :100] = 0
real = R.real_tiles(D / "parity/k2399_d62", 62, 256, 1, seed=2)
if real is None:
    P.partial(f"no rendered stack in {D}/parity/k2399_d62: random tile only")
allt = tiles if real is None else np.concatenate([tiles, real])
ref = R.canonical(R.load_state(WT), allt)
got = P.run_c("canon2", WT, allt, batch=1)
P.compare("canon2 random", ref[:1], got[:1], atol=0.1, mean_tol=0.02)
if real is not None:
    P.compare("canon2 real surface volume", ref[1:], got[1:], atol=0.1, mean_tol=0.02)
print("test_ink_ext_inksdk_canon: OK")

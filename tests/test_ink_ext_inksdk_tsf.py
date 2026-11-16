#!/usr/bin/env python3
"""ufsm ink-ext --model tsf (ink-sdk profiles timesformer@gp-scroll1, @large-scroll1, @scroll5) against tests/inksdk_ref.py timesformer (an
independent torch forward of the lucidrains TimeSformer 0.4.1 divided space-time attention, fp32 CPU) on uint8 tiles 26 x 64 x 64
(clip(0, 200) / 200): random patterns, an edge tile and real 7.91 um surface-volume tiles ($UFSM_INKSDK/parity/k791_d26, rendered by
ink-ext --render-only --dump-layers; PARTIAL without it). Compared: the 64 x 64 logit of the bilinearly upsampled 4 x 4 probability."""
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

D = pathlib.Path(os.environ.get("UFSM_INKSDK", "/vesuvius/ufsm/inksdk")).expanduser()
CASES = [("gp-scroll1", D / "weights/timesformer@gp-scroll1/model.safetensors"), ("large-scroll1", D / "weights/timesformer@large-scroll1/model.safetensors"),
         ("scroll5", D / "weights/timesformer@scroll5/model.ckpt")]
P.need([c[1] for c in CASES])
import numpy as np
import inksdk_ref as R

tiles = P.random_tiles(3, 26, 64, seed=29)
tiles[2, :, :, 40:] = 0
real = R.real_tiles(D / "parity/k791_d26", 26, 64, 4, seed=6)
if real is None:
    P.partial(f"no rendered stack in {D}/parity/k791_d26: random tiles only")
allt = tiles if real is None else np.concatenate([tiles, real])
for tag, w in CASES:
    ref = R.timesformer(R.load_state(w), allt)
    got = P.run_c("tsf", w, allt, batch=4)
    P.compare(f"tsf {tag} random", ref[:3], got[:3], atol=0.05, mean_tol=0.01)
    if real is not None:
        P.compare(f"tsf {tag} real surface volume", ref[3:], got[3:], atol=0.05, mean_tol=0.01)
print("test_ink_ext_inksdk_tsf: OK")

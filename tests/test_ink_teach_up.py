"""ink-teach-up: ink-teach windows at label_scale / up: the mask nearest (255 / 0 exact), teach p bilinear over the old px inside the mask
(renormalised), byte for byte against a numpy reference; json label_scale / up, crop x up, teach_px recounted."""
import json
import pathlib
import subprocess
import tempfile

import importlib.util
import sys

_miss = [m for m in ("numpy", "tifffile", "zarr") if importlib.util.find_spec(m) is None]
if _miss:   # the numpy reference reads the outputs with zarr (the laptop suite 11-22 had none)
    print(f"SKIP: no python {', '.join(_miss)} (the reference needs it)")
    sys.exit(0)

import numpy as np
import tifffile as t
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout
    return r.stdout


def ref(L, M, up):
    H, W = L.shape; H2, W2 = H * up, W * up
    r = np.arange(H2); c = np.arange(W2)
    fy = (r + 0.5) / up - 0.5; fx = (c + 0.5) / up - 0.5
    y0 = np.floor(fy).astype(int); x0 = np.floor(fx).astype(int); ay = fy - y0; ax = fx - x0
    s = np.zeros((H2, W2)); sw = np.zeros((H2, W2))
    for dy in (0, 1):
        for dx in (0, 1):
            yy = np.clip(y0 + dy, 0, H - 1); xx = np.clip(x0 + dx, 0, W - 1)
            g = np.outer(ay if dy else 1 - ay, ax if dx else 1 - ax)
            m = M[np.ix_(yy, xx)] == 255
            s += np.where(m, g * L[np.ix_(yy, xx)], 0); sw += np.where(m, g, 0)
    near = M[np.ix_(r // up, c // up)] == 255
    v = np.where(sw > 0, np.floor(s / np.where(sw > 0, sw, 1) + 0.5), L[np.ix_(r // up, c // up)])
    return np.where(near, v, 0).astype(np.uint8), np.where(near, 255, 0).astype(np.uint8)


rng = np.random.default_rng(3)
H, W = 70, 90
L = rng.integers(0, 256, (H, W)).astype(np.uint8)
M = np.where(rng.random((H, W)) < 0.7, 255, 0).astype(np.uint8); M[:, :10] = 0; L[M == 0] = 0
with tempfile.TemporaryDirectory(prefix="ufsm-teachup-") as tmp:
    d = pathlib.Path(tmp)
    t.imwrite(d / "teach.tif", L); t.imwrite(d / "mask.tif", M)
    (d / "segs.json").write_text(json.dumps({"note": "test", "segments": [{"name": "s1-teach", "tifxyz": "/x/tifxyz", "labels": str(d / "teach.tif"),
                                                                            "mask": str(d / "mask.tif"), "label_scale": 4, "crop": [0, 0, W, H]}]}))
    for up in (2, 3):
        o = d / f"out{up}"
        run(BIN, "ink-teach-up", d / "segs.json", o, "--up", up)
        j = json.loads((o / "segs-teach.json").read_text())["segments"][0]
        L2 = np.asarray(zarr.open(str(o / "s1-teach" / "teach.zarr" / "0"), mode="r")[:]); M2 = np.asarray(zarr.open(str(o / "s1-teach" / "teachmask.zarr" / "0"), mode="r")[:])
        rl, rm = ref(L, M, up)
        assert (M2 == rm).all(), (up, int((M2 != rm).sum()))
        dl = np.abs(L2.astype(int) - rl)   # up 2: exact; up 3: thirds make exact .5 ties whose rounding follows the float sum order (1 LSB)
        assert (dl == 0).all() if up == 2 else (dl.max() <= 1 and (dl > 0).mean() < 1e-3), (up, int((dl > 0).sum()), int(dl.max()))
        assert abs(j["label_scale"] - 4 / up) < 1e-6 and j["crop"] == [0, 0, W * up, H * up] and j["teach_px"] == int((rm == 255).sum())
    r = subprocess.run([str(BIN), "ink-teach-up", str(d / "segs.json"), "rel"], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout
    j = json.loads((d / "rel" / "segs-teach.json").read_text())["segments"][0]   # a relative OUT still writes absolute labels / mask paths
    assert j["labels"].startswith("/") and j["mask"].startswith("/") and pathlib.Path(j["labels"]).parent.is_dir(), j
print("test_ink_teach_up: PASS")

"""img-rankavg: per-input mid-ranks (ties share their mid-rank) over the px valid (> 0) in every input, weighted mean -> 1 + round(253 r),
0 where invalid; byte for byte against a numpy reference on same-size inputs, plus a half-size input resampled onto A's grid
(a constant map: rank 0.5 everywhere it is valid, its no-data quadrant invalid by the nearest px, no 0 blended into its border)."""
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout
    return r.stdout


def midrank(v, ok):
    h = np.bincount(v[ok], minlength=256).astype(np.float64); n = ok.sum()
    c = np.concatenate([[0], np.cumsum(h)[:-1]])
    return ((c + 0.5 * h) / n)[v]


rng = np.random.default_rng(11)
H, W = 96, 128
a = rng.integers(0, 256, (H, W)).astype(np.uint8); b = rng.integers(0, 40, (H, W)).astype(np.uint8)   # b: many ties, some 0
c = rng.integers(1, 256, (H, W)).astype(np.uint8)
with tempfile.TemporaryDirectory(prefix="ufsm-imgrank-") as tmp:
    d = pathlib.Path(tmp)
    for k, im in (("a", a), ("b", b), ("c", c)): t.imwrite(d / f"{k}.tif", im)
    for w in ((1, 1, 1), (2, 0.5, 1)):
        run(BIN, "img-rankavg", d / "o.tif", d / "a.tif", d / "b.tif", d / "c.tif", "--w", ",".join(map(str, w)))
        o = t.imread(d / "o.tif")
        ok = (a > 0) & (b > 0) & (c > 0)
        r = sum(wk * midrank(im, ok) for wk, im in zip(w, (a, b, c))) / sum(w)
        ref = np.where(ok, 1 + np.floor(253 * r + 0.5), 0).astype(np.uint8)   # (lround: halves away from 0, not np.rint's to even)
        assert (o == ref).all(), (w, int((o != ref).sum()))
    s = np.full((H // 2, W // 2), 77, np.uint8); s[: H // 4, : W // 4] = 0   # half size, no-data top-left quadrant
    t.imwrite(d / "s.tif", s)
    run(BIN, "img-rankavg", d / "o2.tif", d / "c.tif", d / "s.tif")
    o = t.imread(d / "o2.tif")
    assert (o[: H // 2 - 2, : W // 2 - 2] == 0).all() and (o[H // 2 + 2:, :] > 0).all()
    ok = o > 0
    ref = (1 + np.floor(253 * (midrank(c, ok) + 0.5) / 2 + 0.5)).astype(np.uint8)
    assert (o[ok] == ref[ok]).all(), int((o[ok] != ref[ok]).sum())
print("test_img_rankavg: PASS")

"""img-gate: a uint8 render x max(gate renders) / 255, byte for byte the numpy float32 step it replaced in scripts/strip/strip_refill.sh
(all 65536 (ink, gate) pairs, deflate and plain inputs, in place and --out)."""
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


rng = np.random.default_rng(7)
H, W = 300, 512   # 153600 px: every (ink, gate) pair at least once, plus random ones
i = np.empty((H, W), np.uint8); g1 = np.empty((H, W), np.uint8); g2 = np.empty((H, W), np.uint8)
a = np.arange(65536); i.flat[:65536] = a >> 8; g1.flat[:65536] = a & 255; g2.flat[:65536] = rng.integers(0, 256, 65536) & (a & 255)
i.flat[65536:] = rng.integers(0, 256, H * W - 65536); g1.flat[65536:] = rng.integers(0, 256, H * W - 65536); g2.flat[65536:] = rng.integers(0, 256, H * W - 65536)
ref = (i.astype(np.float32) * (np.maximum(g1, g2).astype(np.float32) / 255) + 0.5).astype(np.uint8)   # (strip_refill.sh's numpy step until 2026-11-17)
with tempfile.TemporaryDirectory(prefix="ufsm-imggate-") as tmp:
    d = pathlib.Path(tmp)
    t.imwrite(d / "i.tif", i, compression="zlib", rowsperstrip=64); t.imwrite(d / "g1.tif", g1, compression="zlib", rowsperstrip=64); t.imwrite(d / "g2.tif", g2)
    run(BIN, "img-gate", d / "i.tif", d / "g1.tif", d / "g2.tif", "--out", d / "o.tif")
    o = t.imread(d / "o.tif")
    bad = np.flatnonzero(o != ref)
    assert not bad.size, f"{bad.size} px differ, first i {i.flat[bad[0]]} g {max(g1.flat[bad[0]], g2.flat[bad[0]])}: {o.flat[bad[0]]} vs {ref.flat[bad[0]]}"
    run(BIN, "img-gate", d / "i.tif", d / "g1.tif", d / "g2.tif")   # in place
    assert (t.imread(d / "i.tif") == ref).all()
print("img-gate: 65536 pairs equal to numpy")

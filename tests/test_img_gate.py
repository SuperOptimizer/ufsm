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
    # --min T (r3ink.sh's CT papyrus gate): the gate binarised, 255 where max(gates) >= T: IMG kept or zeroed
    t.imwrite(d / "i.tif", i, compression="zlib", rowsperstrip=64)
    for T in (0, 1, 80, 255):
        run(BIN, "img-gate", d / "i.tif", d / "g1.tif", d / "g2.tif", "--min", T, "--out", d / "m.tif")
        assert (t.imread(d / "m.tif") == np.where(np.maximum(g1, g2) >= T, i, 0)).all(), f"--min {T}"
print("img-gate: 65536 pairs equal to numpy, --min binary gate exact")

# img-stretch: strip_refill.sh's gated stretch (numpy 'linear' percentiles of every 7th masked pixel per stripe), several distributions
def np_stretch(ctl, pairs):
    c, k = [], []
    for a, i in pairs: c.append(a[a > 0][::7]); k.append(i[a >= ctl][::7])
    c = np.concatenate(c); k = np.concatenate(k)
    q = lambda v, lo, hi: (int(np.percentile(v, lo)), max(int(np.percentile(v, hi)), int(np.percentile(v, lo)) + 1))
    ca, cb = q(c, 0.5, 99.8); ia, ib = q(k, 1, 99.9); return f"{ca},{cb},{ia},{ib}"
with tempfile.TemporaryDirectory(prefix="ufsm-imgstretch-") as tmp:
    d = pathlib.Path(tmp)
    for trial in range(12):
        pairs, args = [], []
        for s in range(3):
            h, w = int(rng.integers(50, 400)), int(rng.integers(50, 700))
            if trial % 3 == 0: a = rng.integers(0, 256, (h, w)).astype(np.uint8)
            elif trial % 3 == 1: a = np.clip(rng.normal(100, 40, (h, w)), 0, 255).astype(np.uint8)
            else: a = (rng.integers(0, 3, (h, w)) * rng.integers(70, 90, (h, w))).astype(np.uint8)   # few distinct values: interpolation ties
            i = np.clip(rng.normal(60 + 20 * s, 30 + trial, (h, w)), 0, 255).astype(np.uint8)
            t.imwrite(d / f"c{s}.tif", a, compression="zlib" if s else None); t.imwrite(d / f"i{s}.tif", i); pairs.append((a, i)); args += [d / f"c{s}.tif", d / f"i{s}.tif"]
        got = run(BIN, "img-stretch", 80, *args).strip(); ref = np_stretch(80, pairs)
        assert got == ref, f"trial {trial}: img-stretch {got} vs numpy {ref}"
print("img-stretch: 12 trials equal to numpy")

# img-diff: max |a - b| and the differing-byte count, deflate vs plain inputs; a size mismatch fails
with tempfile.TemporaryDirectory(prefix="ufsm-imgdiff-") as tmp:
    d = pathlib.Path(tmp)
    a = rng.integers(0, 256, (257, 257)).astype(np.uint8); b = a.copy(); b[3, 5] = a[3, 5] ^ 1; b[100, 7] = (int(a[100, 7]) + 9) % 256
    t.imwrite(d / "a.tif", a, compression="zlib"); t.imwrite(d / "b.tif", b)
    dd = np.abs(a.astype(int) - b.astype(int))
    assert run(BIN, "img-diff", d / "a.tif", d / "b.tif").strip() == f"max_abs {dd.max()} n_diff {(dd > 0).sum()} n {a.size}"
    assert run(BIN, "img-diff", d / "a.tif", d / "a.tif").strip() == f"max_abs 0 n_diff 0 n {a.size}"
    t.imwrite(d / "c.tif", a[:-1])
    assert subprocess.run([str(BIN), "img-diff", d / "a.tif", d / "c.tif"], capture_output=True).returncode == 1
print("img-diff: exact counts, size mismatch refused")

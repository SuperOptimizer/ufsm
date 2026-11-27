"""ink-score --band-px K (inkeval 2026-11-27): band-tolerant F1 against numpy. A flat tifxyz mesh with canvas = label grid (F 1), blob ink
labels, a map that is the label shifted by 2 px plus noise (the registration offset band-tolerance is for), a mask that drops a block.
Reference: per byte threshold t, precision = predicted px (>= t) within K (Chebyshev) of a scored label-ink px / predicted px, recall =
label-ink px with a predicted px within K / label-ink px, over the scored (mask-on) px only; the best F1 over t. K 0 = the main row's best F1;
the shifted map gains from K 2."""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
G = 72


def boxmax(a, k):   # Chebyshev k-box max with edge clipping
    if k == 0:
        return a.copy()
    p = np.pad(a, k, constant_values=a.min() if a.dtype != bool else False)
    out = np.full_like(a, a.min() if a.dtype != bool else False)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out = np.maximum(out, p[k + dy:k + dy + a.shape[0], k + dx:k + dx + a.shape[1]])
    return out


def band_f1(v, lab, ok, k):
    ink = lab & ok
    near = boxmax(ink, k)
    mx = boxmax(np.where(ok, v.astype(np.int32), -1), k)
    best = 0.0
    for th in range(255, -1, -1):
        pred = ok & (v >= th)
        if not pred.any():
            continue
        pr = (pred & near).sum() / pred.sum()
        re = (ink & (mx >= th)).sum() / ink.sum()
        f = 2 * pr * re / (pr + re) if pr + re > 0 else 0
        best = max(best, f)
    return best


with tempfile.TemporaryDirectory(prefix="ufsm-band-") as tmp:
    d = pathlib.Path(tmp)
    m = d / "seg"; m.mkdir()
    v, u = np.mgrid[0:G, 0:G].astype(np.float32) + 100
    t.imwrite(m / "x.tif", u); t.imwrite(m / "y.tif", v); t.imwrite(m / "z.tif", np.full((G, G), 50, np.float32))
    (m / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "scale": [1.0, 1.0]}))
    rng = np.random.default_rng(11)
    lab = np.zeros((G, G), bool)
    for _ in range(14):   # blob strokes
        y, x = rng.integers(4, G - 8, 2); lab[y:y + rng.integers(2, 6), x:x + rng.integers(3, 9)] = True
    t.imwrite(d / "lab.tif", lab.astype(np.uint8) * 255)
    msk = np.full((G, G), 255, np.uint8); msk[50:66, 3:25] = 0
    t.imwrite(d / "msk.tif", msk)
    sh = np.roll(lab, (2, 1), (0, 1))   # the label 2 px down, 1 right
    a = np.clip(70 + 120 * sh + rng.normal(0, 35, (G, G)), 0, 255).astype(np.uint8)
    t.imwrite(d / "a.tif", a)
    r = subprocess.run([str(BIN), "ink-score", "--surface", str(m), "--labels", str(d / "lab.tif"), "--mask", str(d / "msk.tif"),
                        "--map", f"A={d / 'a.tif'}", "--band-px", "0,2,4", "--row", "T"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = {l.split("\t")[1]: l.split("\t") for l in r.stdout.splitlines()}
    for k in ("A", "A:band0", "A:band2", "A:band4"):
        assert k in rows, (k, r.stdout)
    ok = msk >= 128
    assert abs(float(rows["A:band0"][4]) - float(rows["A"][4])) < 1e-4, (rows["A:band0"], rows["A"])   # K 0 = the plain best F1
    got = {}
    for k in (0, 2, 4):
        w = band_f1(a, lab, ok, k); g = float(rows[f"A:band{k}"][-1]); got[k] = g
        assert abs(g - w) < 2e-4, (k, g, w)
    assert got[2] > got[0] + 0.05, got   # the offset map gains from the band
    r2 = subprocess.run([str(BIN), "ink-score", "--surface", str(m), "--labels", str(d / "lab.tif"), "--map", f"A={d / 'a.tif'}", "--band-px", "x"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert r2.returncode == 2, (r2.returncode, r2.stderr)
    r3 = subprocess.run([str(BIN), "ink-score", "--surface", str(m), "--labels", str(d / "lab.tif"), "--map", f"A={d / 'a.tif'}", "--band-um", "20"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert r3.returncode == 2 and "unknown" in r3.stderr, (r3.returncode, r3.stderr)   # no voxel size: refused, not guessed
    r4 = subprocess.run([str(BIN), "ink-score", "--surface", str(m), "--labels", str(d / "lab.tif"), "--mask", str(d / "msk.tif"), "--map", f"A={d / 'a.tif'}",
                         "--surf-um", "10", "--band-um", "20,41", "--row", "T"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert r4.returncode == 0, r4.stderr
    rows4 = {l.split("\t")[1]: l.split("\t") for l in r4.stdout.splitlines()}
    assert abs(float(rows4["A:band20um"][-1]) - got[2]) < 1e-4 and abs(float(rows4["A:band41um"][-1]) - got[4]) < 1e-4, rows4   # 20 / 41 um at 10 um px = 2 / 4 px
    print("test_ink_band: OK", {k: round(v, 4) for k, v in got.items()})

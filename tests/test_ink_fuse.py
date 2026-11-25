"""ink-score --fuse (2026-11-24, output fusion of reader3 + 3-D nets): a flat tifxyz mesh with canvas = label grid (F 1), two uint8 maps
carrying noisy ink signal, a mask that drops a block, one map that does not cover the last columns. Checked against numpy: the common pixel
set (mask on, covered by both maps), each input's AUC there (NAME@c), fuse-mean = the weighted mean of the bytes (--weights), fuse-rank =
the weighted mean of per-map normalised ranks (ties averaged); AUC tie-aware (Mann-Whitney, average ranks)."""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
G = 64


def ranks(x):   # average ranks (1-based) with ties averaged
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(cnt); start = end - cnt + 1
    return ((start + end) / 2.0)[inv]


def auc(v, y):
    r = ranks(v); ni = int(y.sum()); nn = len(y) - ni
    return (r[y].sum() - ni * (ni + 1) / 2) / (ni * nn)


with tempfile.TemporaryDirectory(prefix="ufsm-fuse-") as tmp:
    d = pathlib.Path(tmp)
    m = d / "seg"; m.mkdir()
    v, u = np.mgrid[0:G, 0:G].astype(np.float32) + 100
    t.imwrite(m / "x.tif", u); t.imwrite(m / "y.tif", v); t.imwrite(m / "z.tif", np.full((G, G), 50, np.float32))
    (m / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "scale": [1.0, 1.0]}))
    rng = np.random.default_rng(7)
    lab = (rng.random((G, G)) < 0.3)
    t.imwrite(d / "lab.tif", lab.astype(np.uint8) * 255)
    msk = np.full((G, G), 255, np.uint8); msk[:10, :20] = 0
    t.imwrite(d / "msk.tif", msk)
    a = np.clip(80 + 60 * lab + rng.normal(0, 50, (G, G)), 0, 255).astype(np.uint8)
    b = np.clip(100 + 40 * lab + rng.normal(0, 40, (G, G)), 0, 255).astype(np.uint8)[:, :G - 6]   # 6 columns short: not covered
    t.imwrite(d / "a.tif", a); t.imwrite(d / "b.tif", b)
    r = subprocess.run([str(BIN), "ink-score", "--surface", str(m), "--labels", str(d / "lab.tif"), "--mask", str(d / "msk.tif"),
                        "--map", f"A={d / 'a.tif'}", "--map", f"B={d / 'b.tif'}", "--fuse", "mean,rank", "--weights", "0.7,0.3", "--row", "T"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = {l.split("\t")[1]: l.split("\t") for l in r.stdout.splitlines()}
    for k in ("A", "B", "A@c", "B@c", "fuse-mean", "fuse-rank"):
        assert k in rows, (k, r.stdout)
    ok = (msk >= 128); ok[:, G - 6:] = False   # bilinear sampling needs cx <= mw - 1: the last covered column is G - 7
    y = lab[ok]; va = a[ok].astype(float); vb = np.zeros((G, G)); vb[:, :G - 6] = b; vb = vb[ok]
    assert int(rows["fuse-mean"][2]) == ok.sum(), (rows["fuse-mean"][2], ok.sum())
    want = {"A@c": auc(va, y), "B@c": auc(vb, y), "fuse-mean": auc((0.7 * va + 0.3 * vb).astype(np.float32), y)}
    qa = (ranks(va) - 1) / (len(va) - 1); qb = (ranks(vb) - 1) / (len(vb) - 1)
    want["fuse-rank"] = auc((0.7 * qa.astype(np.float32) + 0.3 * qb.astype(np.float32)).astype(np.float32), y)
    for k, w in want.items():
        got = float(rows[k][-1])
        assert abs(got - w) < 2e-4, (k, got, w)
    assert abs(float(rows["A"][-1]) - auc(a[msk >= 128].astype(float), lab[msk >= 128])) < 2e-4   # the per-map row: its own pixels
    print("test_ink_fuse: OK", {k: round(float(rows[k][-1]), 4) for k in rows})

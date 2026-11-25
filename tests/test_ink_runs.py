"""ink-runs + ink-score --edge (2026-11-25, the 116 keV diagnosis). ink-runs: vertical ink bars of known widths (5 and 9 px, 40 px tall)
on a 64 x 96 label image -> the per-pixel width min(row run, column run) quartiles; a bar touching the image border is left out; the mask
only selects pixels. ink-score --edge 1: a map that is bright only in the bars' interior columns -> NAME:core AUC 1 (interior ink vs every
negative), NAME:edge AUC 0.5 for edge ink at the negatives' level; checked against numpy."""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
H, W = 64, 96


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout


with tempfile.TemporaryDirectory(prefix="ufsm-runs-") as tmp:
    d = pathlib.Path(tmp)
    lab = np.zeros((H, W), np.uint8)
    lab[10:50, 10:15] = 255; lab[10:50, 30:39] = 255; lab[10:50, 0:4] = 255   # widths 5, 9; the last touches the border
    t.imwrite(d / "lab.tif", lab)
    msk = np.full((H, W), 255, np.uint8); msk[:, 60:] = 0
    t.imwrite(d / "msk.tif", msk)
    o = run(BIN, "ink-runs", d / "lab.tif", "--mask", d / "msk.tif", "--px-um", "2", "--row", "T").split("\t")
    n, p25, p50, p75 = int(o[1]), float(o[2]), float(o[3]), float(o[4])
    assert n == 40 * 5 + 40 * 9, o          # the border bar is open; every pixel of the closed bars has width = its bar's width (<= 40)
    w = np.sort(np.r_[np.full(200, 5), np.full(360, 9)])
    assert (p25, p50, p75) == (w[n // 4], w[n // 2], w[3 * n // 4]), o
    assert float(o[6]) == p50 * 2, o
    # --edge 1 on a flat mesh (canvas = label grid)
    m = d / "seg"; m.mkdir()
    v, u = np.mgrid[0:H, 0:W].astype(np.float32) + 100
    t.imwrite(m / "x.tif", u); t.imwrite(m / "y.tif", v); t.imwrite(m / "z.tif", np.full((H, W), 50, np.float32))
    (m / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "scale": [1.0, 1.0]}))
    ink = lab >= 128
    pad = np.pad(~ink, 1, constant_values=False)
    near0 = np.zeros_like(ink)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near0 |= pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    edge = ink & near0; core = ink & ~near0
    mp = np.full((H, W), 50, np.uint8); mp[core] = 200
    t.imwrite(d / "map.tif", mp)
    rows = {l.split("\t")[1]: l.split("\t") for l in run(BIN, "ink-score", "--surface", m, "--labels", d / "lab.tif", "--mask", d / "msk.tif",
                                                         "--map", f"M={d / 'map.tif'}", "--edge", "1", "--row", "T").splitlines()}
    sup = msk >= 128
    assert int(rows["M:core"][2]) == (core & sup).sum() + (~ink & sup).sum(), rows["M:core"]
    assert int(rows["M:edge"][2]) == (edge & sup).sum() + (~ink & sup).sum(), rows["M:edge"]
    assert abs(float(rows["M:core"][-1]) - 1.0) < 1e-9 and abs(float(rows["M:edge"][-1]) - 0.5) < 1e-9, rows
    print("test_ink_runs: OK", o[1:5], rows["M:core"][-1], rows["M:edge"][-1])

"""ink-runs + ink-score --edge (2026-11-25, the 116 keV diagnosis). ink-runs: vertical ink bars of known widths (5 and 9 px, 40 px tall)
on a 64 x 96 label image -> the per-pixel width min(row run, column run) quartiles; a bar touching the image border is left out; the mask
only selects pixels. ink-score --edge 1: a map that is bright only in the bars' interior columns -> NAME:core AUC 1 (interior ink vs every
negative), NAME:edge AUC 0.5 for edge ink at the negatives' level; checked against numpy."""
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("UFSM_BIN", ROOT / "build/ufsm"))
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
    # --quality (TTA goals beyond AUC): a noisy map, every metric against numpy
    rng = np.random.default_rng(3)
    mq = np.clip(rng.normal(70, 12, (H, W)) + 120 * ink, 0, 255).astype(np.uint8); mq[ink & (rng.random((H, W)) < 0.2)] = 60
    t.imwrite(d / "mq.tif", mq)
    q = [l.split("\t") for l in run(BIN, "ink-score", "--surface", m, "--labels", d / "lab.tif", "--mask", d / "msk.tif", "--map", f"Q={d / 'mq.tif'}",
                                    "--quality", "--quality-far", "8", "--row", "T").splitlines() if l.split("\t")[1] == "Q:q"][0]
    V = mq.astype(np.float64) / 255; I = ink & sup; B = ~ink & sup
    dil = np.zeros_like(ink); pi = np.pad(ink, 8)
    for dy in range(-8, 9):
        for dx in range(-8, 9): dil |= pi[8 + dy:8 + dy + H, 8 + dx:8 + dx + W]
    Fr = ~sup & ~dil
    def pct(x): b = np.sort(np.round(x * 255)); return b[int(np.floor(0.95 * (len(b) - 1)))] / 255
    def speck(S):
        out = []
        for r, c in zip(*np.nonzero(S)):
            w = V[max(r - 4, 0):r + 5, max(c - 4, 0):c + 5][S[max(r - 4, 0):r + 5, max(c - 4, 0):c + 5]]
            if len(w) >= 2: out.append(np.sqrt(max(np.mean(w * w) - np.mean(w) ** 2, 0)))
        return np.mean(out)
    def recall(f):
        hb = np.bincount(mq[B], minlength=256); hi = np.bincount(mq[I], minlength=256); nb = B.sum(); fp = tp = 0.0
        for b in range(255, -1, -1):
            if (fp + hb[b]) / nb >= f: tp += ((f * nb - fp) / hb[b] if hb[b] else 0) * hi[b]; break
            fp += hb[b]; tp += hi[b]
        return tp / I.sum()
    g = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if I[r, c] and not (ink[r - 1, c] and ink[r + 1, c] and ink[r, c - 1] and ink[r, c + 1]):
                g.append(np.hypot(0.5 * (V[r, c + 1] - V[r, c - 1]), 0.5 * (V[r + 1, c] - V[r - 1, c])))
    dm = V[I].mean() - V[B].mean()
    ref = [I.sum(), B.sum(), V[B].mean(), pct(V[B]), (mq[B] >= 128).mean(), speck(B), Fr.sum(), V[Fr].mean(), pct(V[Fr]), (mq[Fr] >= 128).mean(), speck(Fr),
           dm, dm / np.sqrt(0.5 * (V[I].var() + V[B].var())), recall(0.01), recall(0.05), np.mean(g)]
    for k, (a, b) in enumerate(zip(q[2:], ref)): assert abs(float(a) - b) < 1.5e-4, (k, a, b, q)
    print("test_ink_runs: OK", o[1:5], rows["M:core"][-1], rows["M:edge"][-1], "quality", q[2:])

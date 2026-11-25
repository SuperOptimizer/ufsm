"""ink-devcrops (2026-11-24, the 8.6-9.4 um ink dev set): scoring crops of the windows inside held-out boxes. A flat tifxyz mesh on an
8.64 um grid (scale 0.05: 20 voxels per cell, canvas 1 px per voxel), labels at label_scale 1.25 (1.25 canvas px per label px) with a
"crop" window, ink stripes in voxel space. Checked: only windows touching the box are cut, cells outside the box masked out; the crop
grid (scale 0.25) reproduces the mesh; labels.tif equals the stripes on the canvas through F and the crop window; supervised px / ink
share / box_l0 in crop.json and crops.tsv; the --map-boxes identity check is in the suite's reg tests (not here)."""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
G = 41                    # grid points per side (20 voxels apart -> canvas 800 px)
F = 1.25                  # canvas px per label px
LW = 640                  # label px per side (800 / 1.25)
WIN = 128


def run(*a, ok=True):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    assert (r.returncode == 0) == ok, r.stdout
    return r.stdout


with tempfile.TemporaryDirectory(prefix="ufsm-devcrops-") as tmp:
    d = pathlib.Path(tmp)
    m = d / "A/tifxyz"; m.mkdir(parents=True)
    v, u = np.mgrid[0:G, 0:G].astype(np.float32) * 20 + 1000          # rows = y, cols = x (level-0 voxels); canvas px (cu, cv) = voxel (1000 + cu, 1000 + cv)
    t.imwrite(m / "x.tif", u); t.imwrite(m / "y.tif", v); t.imwrite(m / "z.tif", np.full((G, G), 500, np.float32))
    (m / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "scale": [0.05, 0.05], "voxel_size_um": 8.64}))
    lc = (np.arange(LW) // 10 % 2).astype(np.uint8)                   # label stripes: ink where (label col // 10) is odd
    t.imwrite(d / "A/lab.tif", np.broadcast_to(lc[None, :], (LW, LW)) * 255)
    t.imwrite(d / "A/msk.tif", np.full((LW, LW), 1, np.uint8))         # a 0/1 mask: read as 0/255
    crop = [0, 0, LW, 400]                                             # label rows >= 400 unsupervised (canvas rows >= 500)
    (d / "segs.json").write_text(json.dumps({"segments": [{"name": "A", "tifxyz": str(m), "labels": str(d / "A/lab.tif"), "mask": str(d / "A/msk.tif"),
                                                           "label_scale": F, "crop": crop}]}))
    box = [0, 1000, 1200, 1000, 400, 300]                              # z, y, x, nz, ny, nx: canvas rows 0..399, cols 200..499
    out = run(BIN, "ink-devcrops", d / "segs.json", "--box", ",".join(map(str, box)), "--ct-key", "S/scan.zarr", "--um", 8.64, "--out", d / "c",
              "--win", WIN, "--min-sup", 0.05, "--min-ink", 10, "--scroll", "S", "--tag", "T")
    rows = [l.split("\t") for l in (d / "c/crops.tsv").read_text().splitlines() if not l.startswith("#")]

    def expect(v0, u0):   # the supervised px of a window: cells whose 4 corners lie in the box (canvas cols 200..499, rows 0..399), label rows < 400
        cv, cu = np.mgrid[v0:v0 + WIN, u0:u0 + WIN]
        cell_in = ((cu // 4 * 4) >= 200) & ((cu // 4 * 4 + 4) <= 499) & ((cv // 4 * 4 + 4) <= 399)
        lr, lcol = np.floor((cv + 0.5) / F).astype(int), np.floor((cu + 0.5) / F).astype(int)
        return cell_in & (lr < 400), lcol
    want = []
    for a in range(800 // WIN):
        for b in range(800 // WIN):
            sup, lcol = expect(a * WIN, b * WIN); ink = int(((lcol[sup] // 10) % 2 == 1).sum())
            if sup.sum() >= 0.05 * WIN * WIN and ink >= 10 and sup.sum() - ink >= 10: want.append(f"T_s00_{a}_{b}")
    assert sorted(r[0] for r in rows) == sorted(want) and len(want) >= 8, (sorted(r[0] for r in rows), want, out)
    for r in rows:
        c = d / "c" / r[0]; a, b = map(int, r[0].split("_")[-2:]); v0, u0 = a * WIN, b * WIN
        X, Y, Z = (t.imread(c / f"{k}.tif") for k in "xyz")
        assert X.shape == (WIN // 4 + 1, WIN // 4 + 1), X.shape
        assert np.allclose(X, 1000 + u0 + 4 * np.arange(WIN // 4 + 1)[None, :], atol=1e-3) and np.allclose(Y, 1000 + v0 + 4 * np.arange(WIN // 4 + 1)[:, None], atol=1e-3)
        assert json.loads((c / "meta.json").read_text())["scale"][0] == 0.25
        L, M = t.imread(c / "labels.tif"), t.imread(c / "mask.tif")
        sup, lcol = expect(v0, u0)
        assert np.array_equal(M == 255, sup), (r[0], int((M == 255).sum()), int(sup.sum()))
        assert np.array_equal(L[sup] == 255, (lcol[sup] // 10 % 2) == 1), r[0]
        assert not L[~sup].any()
        j = json.loads((c / "crop.json").read_text())
        assert j["supervised_px"] == int(sup.sum()) == int(r[2]) and abs(j["label_scale"] - F) < 1e-9, j
        assert abs(j["ink_share"] - (L[sup] == 255).mean()) < 1e-3, j
        bz, by, bx, nz, ny, nx = j["box_l0"]
        assert bz <= 500 - 32 and bz + nz >= 500 + 32 and by <= 1000 + v0 and bx <= 1000 + max(u0, 200) and bx % 4 == 0, j
    # --exclude (probe training crops outside held-out boxes): the whole canvas as --box, the box above excluded: exactly the windows
    # that would qualify without it and have no valid vertex inside the excluded box remain
    big = "0,0,0,100000,100000,100000"
    run(BIN, "ink-devcrops", d / "segs.json", "--box", big, "--ct-key", "S/scan.zarr", "--um", 8.64, "--out", d / "x0", "--win", WIN, "--min-sup", 0.05,
        "--min-ink", 10, "--tag", "X")
    run(BIN, "ink-devcrops", d / "segs.json", "--box", big, "--exclude", ",".join(map(str, box)), "--ct-key", "S/scan.zarr", "--um", 8.64, "--out", d / "x1",
        "--win", WIN, "--min-sup", 0.05, "--min-ink", 10, "--tag", "X")
    all_ = [l.split("\t")[0] for l in (d / "x0/crops.tsv").read_text().splitlines() if not l.startswith("#")]
    kept = [l.split("\t")[0] for l in (d / "x1/crops.tsv").read_text().splitlines() if not l.startswith("#")]
    def touches(name):   # a window's vertices (canvas u0 + 4 k, v0 + 4 k, k 0..WIN/4) against the box (canvas cols 200..499, rows 0..399)
        a, b = map(int, name.split("_")[-2:]); cu = b * WIN + 4 * np.arange(WIN // 4 + 1); cv = a * WIN + 4 * np.arange(WIN // 4 + 1)
        return bool(((cu >= 200) & (cu < 500)).any() and ((cv >= 0) & (cv < 400)).any())
    assert sorted(kept) == sorted(n for n in all_ if not touches(n)) and 0 < len(kept) < len(all_), (kept, all_)
    run(BIN, "ink-devcrops", d / "segs.json", "--box", big, "--exclude", "1,2,3", "--ct-key", "S/scan.zarr", "--um", 8.64, "--out", d / "x2", ok=False)
    # a box no window touches: no crops, exit 0
    run(BIN, "ink-devcrops", d / "segs.json", "--box", "0,5000,5000,10,10,10", "--ct-key", "S/scan.zarr", "--um", 8.64, "--out", d / "e")
    assert len((d / "e/crops.tsv").read_text().splitlines()) == 1
    run(BIN, "ink-devcrops", d / "segs.json", "--ct-key", "S/scan.zarr", "--um", 8.64, "--out", d / "f", ok=False)   # no --box: usage
    # --ctx-of (reader3 TTA with context): PAD 0 reproduces a crop's x / y / z byte for byte; PAD 16 is the window grown by 16 canvas px
    # per side (4 grid cells), the crop its interior, vertices before canvas 0 invalid (-1)
    c0 = sorted(r[0] for r in rows)[0]; cd = d / "c" / c0; a, b = map(int, c0.split("_")[-2:]); v0, u0 = a * WIN, b * WIN
    run(BIN, "ink-devcrops", d / "segs.json", "--ctx-of", cd, "--pad", 0, "--out", d / "k0")
    for k in "xyz":
        assert (d / "k0" / f"{k}.tif").read_bytes() == (cd / f"{k}.tif").read_bytes() or np.array_equal(t.imread(d / "k0" / f"{k}.tif"), t.imread(cd / f"{k}.tif")), k
    run(BIN, "ink-devcrops", d / "segs.json", "--ctx-of", cd, "--pad", 16, "--out", d / "k16")
    X16 = t.imread(d / "k16/x.tif"); n = WIN // 4 + 1
    assert X16.shape == (n + 8, n + 8) and np.array_equal(X16[4:4 + n, 4:4 + n], t.imread(cd / "x.tif")), X16.shape
    if u0 == 0: assert (X16[:, :4] == -1).all()
    run(BIN, "ink-devcrops", d / "segs.json", "--ctx-of", cd, "--pad", 6, "--out", d / "k6", ok=False)   # pad not a multiple of 4
print("test_ink_devcrops: OK")

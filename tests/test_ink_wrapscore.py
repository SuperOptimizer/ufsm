"""ink-wrapscore (src/wrapscore.c) against its Python reference scripts/inkgt/wrapscore.py on a synthetic unroll: one published segment
(a tilted plane, labels / mask / validation rasters, listed twice: whole and a crop, which dedups), two wraps (w001 on the segment's plane
at another grid, scale and voxel size, slightly bent so the foot points need the Gauss-Newton steps; w002 far away), ink maps rendered on
w001's canvas. Checks: per split (heldout, rest, all) the sample count and the AUC equal wrapscore.py's at tol 40 um (wrapscore's
INK >= 0 counts: --nonzero none); a reference pair's delta is the AUC difference; the cached --table gives the same tsv; --tiles-out
lists tiles; hard failures (exit 1, no tsv) on a missing map dir, a --need wrap without a map, an unreadable segment."""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
REF = ROOT / "scripts/inkgt/wrapscore.py"


def run(*a, ok=True):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if ok:
        assert r.returncode == 0, r.stdout
    return r


def tsv(p):
    L = [l.split("\t") for l in pathlib.Path(p).read_text().strip().split("\n")]
    return {r[0]: dict(zip(L[0], r)) for r in L[1:]}


def tifxyz(d, X, Y, Z, scale, um):
    d.mkdir(parents=True)
    for n, a in (("x", X), ("y", Y), ("z", Z)):
        t.imwrite(d / f"{n}.tif", a.astype(np.float32))
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [scale, scale], "voxel_size_um": um}))


rng = np.random.default_rng(3)
with tempfile.TemporaryDirectory(prefix="ufsm-wrapscore-") as tmp:
    d = pathlib.Path(tmp)
    # the published segment: grid 97 x 129 at scale 0.25 (canvas 384 x 512 px, 1 px = 1 level-0 voxel of 2.399 um), a tilted plane
    gh, gw = 97, 129; v, u = np.mgrid[0:gh, 0:gw].astype(np.float64)
    Y0, X0 = 600 + 4 * v, 700 + 4 * u; Z0 = 2000 + 0.3 * Y0 - 0.2 * X0
    Z0[:3, :5] = -1   # a grid hole: no samples there
    tifxyz(d / "seg/tifxyz", X0, Y0, Z0, 0.25, 2.399)
    H, W = 384, 512; yy, xx = np.mgrid[0:H, 0:W]
    lab = (((np.sin(yy / 9.0) + np.cos(xx / 13.0)) > 0.6) * 255).astype(np.uint8)
    mask = np.full((H, W), 255, np.uint8); mask[300:, 400:] = 0
    val = np.zeros((H, W), np.uint8); val[:150, :200] = 255
    for n, a in (("labels", lab), ("mask", mask), ("val", val)):
        t.imwrite(d / f"seg/{n}.tif", a)
    s = str(d / "seg")
    seg = {"name": "seg1", "tifxyz": s + "/tifxyz", "labels": s + "/labels.tif", "mask": s + "/mask.tif", "validation": s + "/val.tif"}
    (d / "segs.json").write_text(json.dumps({"segments": [seg, dict(seg, crop=[0, 0, 512, 384]), dict(seg, name="seg1b", crop=[64, 200, 256, 128])]}))
    # a holdout box of this scan, one of another scan (ignored)
    (d / "hold.json").write_text(json.dumps({"boxes": [{"name": "A", "ct": "S/scan.zarr", "l0": [0, 1000, 1100, 4000, 400, 300]},
                                                        {"name": "B", "ct": "S/other.zarr", "l0": [0, 0, 0, 9000, 9000, 9000]}]}))
    # wraps in 9.596 um voxels (x 1/4), grid step 2 wrap voxels (scale 0.5): w001 on the plane with a small bend and offset, w002 far away
    wh, ww = 51, 67; v, u = np.mgrid[0:wh, 0:ww].astype(np.float64)
    Yw = (590 + 8.1 * v) / 4; Xw = (690 + 8.05 * u + 3 * np.sin(v / 7)) / 4; Zw = (2000 + 0.3 * (Yw * 4) - 0.2 * (Xw * 4) + 1.5 * np.cos(u / 5)) / 4
    tifxyz(d / "wraps/w001", Xw, Yw, Zw, 0.5, 9.596)
    tifxyz(d / "wraps/w002", Xw, Yw, Zw + 500, 0.5, 9.596)
    # maps on w001's render canvas (render_out_px(n, 2) = 2 (n - 1) + 1): the label read back through the wrap's geometry, plus noise
    oh, ow = 2 * (wh - 1) + 1, 2 * (ww - 1) + 1; py, px = np.mgrid[0:oh, 0:ow] / 2.0
    ly = np.clip(np.rint((590 + 8.1 * py) - 600), 0, H - 1).astype(int); lx = np.clip(np.rint((690 + 8.05 * px) - 700), 0, W - 1).astype(int)
    good = np.clip(lab[ly, lx] * 0.5 + rng.integers(0, 128, (oh, ow)), 1, 255).astype(np.uint8)
    (d / "mA").mkdir(); (d / "mB").mkdir(); t.imwrite(d / "mA/w001.tif", good); t.imwrite(d / "mB/w001.tif", rng.integers(1, 256, (oh, ow)).astype(np.uint8))
    # the reference
    run("python3", REF, d / "wraps", d / "segs.json", d / "ref.tsv", "--map", d / "mA", "--holdouts", d / "hold.json", "--ct-key", "X/scan.zarr", "--tol", "40")
    R = [l.split("\t") for l in (d / "ref.tsv").read_text().strip().split("\n")]; R = {(r[0], r[1]): dict(zip(R[0], r)) for r in R[1:]}
    common = [BIN, "ink-wrapscore", d / "wraps", d / "segs.json", d / "out", "--ct-key", "X/scan.zarr", "--holdouts", d / "hold.json", "--tol", "40", "--nonzero", "none", "--min-n", "10"]
    o = run(*common, "--maps", f"A={d / 'mA'},B={d / 'mB'}", "--ref", "B", "--split", "heldout,rest,all", "--table", d / "tab.uws").stdout
    for sp in ("heldout", "rest", "all"):
        r, c = R[("40", sp)], tsv(d / f"out.{sp}.tsv")
        assert int(c["A"]["n"]) == int(r["covered"]) and int(r["covered"]) > 100, (sp, c["A"], r, o)
        assert abs(float(c["A"]["auc"]) - float(r["ink_auc"])) < 1e-4, (sp, c["A"], r)
        assert abs(float(c["A"]["delta_vs_ref"]) - (float(c["A"]["auc"]) - float(c["B"]["auc"]))) < 2e-4, c
        assert float(c["A"]["auc"]) > 0.8 and abs(float(c["B"]["auc"]) - 0.5) < 0.1, c
    first = (d / "out.heldout.tsv").read_text()
    o2 = run(*common, "--maps", f"A={d / 'mA'},B={d / 'mB'}", "--ref", "B", "--split", "heldout", "--table", d / "tab.uws").stdout
    assert "table:" in o2 and (d / "out.heldout.tsv").read_text() == first, o2
    o3 = run(*common, "--split", "heldout", "--table", d / "tab.uws", "--need", "w001", "--tiles-out", d / "tiles").stdout
    assert (d / "tiles/tiles0.txt").read_text().strip() and (d / "tiles/tiles256.txt").read_text().strip(), o3
    # hard failures: no tsv written
    (d / "out.heldout.tsv").unlink()
    for extra in (["--maps", f"A={d / 'nodir'}"], ["--maps", f"A={d / 'mA'}", "--need", "w001,w002"]):
        r = run(*common, *extra, "--split", "heldout", "--table", d / "tab.uws", ok=False)
        assert r.returncode == 1 and not (d / "out.heldout.tsv").exists(), (extra, r.stdout)
    (d / "segs2.json").write_text(json.dumps({"segments": [dict(seg, labels=s + "/missing.tif")]}))
    r = run(*common[:3], d / "segs2.json", *common[4:], "--maps", f"A={d / 'mA'}", ok=False)
    assert r.returncode == 1 and "missing.tif" in r.stdout, r.stdout
print("test_ink_wrapscore: ok")

"""reg tifxyz: a tifxyz on the fine scan moved into the coarse scan's L0 voxels through a reg map (affine A, no field: f2c = A p), invalid
vertices kept invalid, meta scale x div (coarse um / fine um); --c2f back again within float32 rounding; the per-vertex values equal
`reg map --f2c` on the same points."""
import json
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


rng = np.random.default_rng(5)
th = 0.3; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) / 3.9025   # fine 2.399 -> coarse 9.362
A = np.hstack([R, [[120.5], [-33.25], [7.0]]])
H, W = 40, 60
P = np.stack([rng.uniform(1000, 9000, (H, W)), rng.uniform(1000, 9000, (H, W)), rng.uniform(100, 20000, (H, W))], -1).astype(np.float32)
bad = rng.random((H, W)) < 0.1; P[bad] = -1
with tempfile.TemporaryDirectory(prefix="ufsm-regtfx-") as tmp:
    d = pathlib.Path(tmp); s = d / "seg1" / "tifxyz"; s.mkdir(parents=True)
    for k, c in enumerate("xyz"): t.imwrite(s / f"{c}.tif", P[..., k])
    (s / "meta.json").write_text(json.dumps({"format": "tifxyz", "scale": [0.05, 0.05], "type": "seg", "uuid": "out"}, indent=4))
    (d / "reg.json").write_text(json.dumps({"name": "t", "coarse": {"root": "x", "key": "c", "um": 9.362}, "fine": {"root": "x", "key": "f", "um": 2.399},
                                            "A": A.tolist()}))
    run(BIN, "reg", "tifxyz", d / "reg.json", d / "out", s)
    Q = np.stack([t.imread(d / "out" / "seg1" / f"{c}.tif") for c in "xyz"], -1)
    assert (Q[bad] == -1).all()
    ref = (P[~bad].astype(np.float64) @ R.T + A[:, 3]).astype(np.float32)
    assert np.abs(Q[~bad] - ref).max() < 1e-3, np.abs(Q[~bad] - ref).max()
    m = json.loads((d / "out" / "seg1" / "meta.json").read_text())
    assert abs(m["scale"][0] - 0.05 * 9.362 / 2.399) < 1e-9, m["scale"]
    pts = [",".join("%.6f" % v for v in P[y, x]) for y, x in np.argwhere(~bad)[:20]]
    got = np.array([[float(v) for v in ln.split(",")] for ln in run(BIN, "reg", "map", d / "reg.json", "--f2c", *pts).split()])
    assert np.abs(got - Q[~bad][:20]).max() < 2e-3
    run(BIN, "reg", "tifxyz", d / "reg.json", d / "back", d / "out" / "seg1", "--c2f")
    B = np.stack([t.imread(d / "back" / "seg1" / f"{c}.tif") for c in "xyz"], -1)
    assert (B[bad] == -1).all() and np.abs(B[~bad] - P[~bad]).max() < 5e-3
print("test_reg_tifxyz: PASS")

"""ink-ext / tifxyz-crop: a tifxyz whose meta.json has no voxel_size_um is refused unless --surf-um (or, for ink-ext, an explicit --ct-um)
gives it; no silent 2.399 default (lead 11-26: the HercUeNet 1203 mesh, 9.362 um voxels, was read at x 0.256). ink-ext refuses before the
GPU / weights / CT are touched (run here with no GPU and no weights); tifxyz-crop carries the given size into the crop's meta."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

if importlib.util.find_spec("tifffile") is None or importlib.util.find_spec("numpy") is None:
    print("SKIP: no python numpy / tifffile")
    sys.exit(0)

import numpy as np
import tifffile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")
REFUSE = "has no voxel_size_um"


def mesh(d, vox):
    d.mkdir(parents=True)
    r, c = np.meshgrid(np.arange(6, dtype=np.float32), np.arange(5, dtype=np.float32), indexing="ij")
    for n, a in (("x", 100 + 2 * c), ("y", 200 + 2 * r), ("z", np.full_like(r, 300))):
        tifffile.imwrite(d / f"{n}.tif", a.astype(np.float32))
    m = {"format": "tifxyz", "type": "seg", "uuid": d.name, "scale": [0.5, 0.5]}
    if vox:
        m["voxel_size_um"] = vox
    (d / "meta.json").write_text(json.dumps(m))
    return d


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60, env=ENV)
    return r.returncode, r.stdout


def ink_ext(s, *extra):
    return run(BIN, "ink-ext", "--model", "reader3", "--weights", "/nonexistent/w.pth", "--surface", s, "--ct", "/nonexistent", "K",
               "--out", s.parent / "o", *extra)


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    bare, good = mesh(td / "bare", None), mesh(td / "good", 9.362)

    rc, out = ink_ext(bare)
    assert rc == 2 and REFUSE in out, (rc, out)
    for extra in (("--surf-um", "9.362"), ("--ct-um", "9.362")):   # given: past the check (then fails on the missing GPU / weights, not on it)
        rc, out = ink_ext(bare, *extra)
        assert REFUSE not in out and rc != 0, (extra, rc, out)
    rc, out = ink_ext(good)
    assert REFUSE not in out, out

    rc, out = run(BIN, "tifxyz-crop", bare, td / "c0", "0,0,3,3")
    assert rc == 2 and REFUSE in out and not (td / "c0/meta.json").exists(), (rc, out)
    rc, out = run(BIN, "tifxyz-crop", bare, td / "c1", "0,0,3,3", "--surf-um", "9.362")
    assert rc == 0, out
    assert abs(json.loads((td / "c1/meta.json").read_text())["voxel_size_um"] - 9.362) < 1e-6
    rc, out = run(BIN, "tifxyz-crop", good, td / "c2", "0,0,3,3")
    assert rc == 0 and abs(json.loads((td / "c2/meta.json").read_text())["voxel_size_um"] - 9.362) < 1e-6, out
print("PASS test_ink_ext_surfum")

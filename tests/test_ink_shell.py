"""ink-ext --shell-list / --shell-prev / --ct-shell (inkeval 2026-11-27, the q2 CT shell for coarse-scan ink). CPU only: --shell-list and
--render-only load no net and touch no GPU. A two-shard (z) CT whose main store is 100 everywhere and whose shell copy is 200 everywhere;
the shell's coverage is what its shell.boxes says. Checked:
- --shell-list of surface A (z 100..599): one box per 1024^3 shard, only shard (0, 0, 0), header "# level 0 shard 1024", chunk-aligned and
  covering A's rows; --shell-prev with surface B (z 100..1199): shard 0 grown (named in .redo), shard 1 added (not in .redo);
- --render-only --dump-layers of B through --ct-shell with A's boxes: rows (z) inside A's chunks read 200, rows past them 100, and the
  DEFAULTED line counts the chunks read from the main store; with B's boxes: all 200, no DEFAULTED; a shell root without shell.boxes:
  DEFAULTED, all 100 (the q8 fallback)."""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = str(ROOT / "build/ufsm")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")
SH = (1280, 256, 128)   # (two 1024^3 shards along z: the shard key of every axis is exercised)
YS = 150.0


def run(*a, ok=True):
    r = subprocess.run([str(x) for x in a], capture_output=True, text=True, timeout=300, env=ENV)
    if ok:
        assert r.returncode == 0, (a, r.stdout[-2000:], r.stderr[-4000:])
    return r


def mesh(d, y0, y1):
    d.mkdir(parents=True)
    z, x = np.mgrid[y0:y1, 20:100].astype(np.float32)   # a plane y = YS, grid rows along z
    t.imwrite(d / "x.tif", x); t.imwrite(d / "y.tif", np.full(x.shape, YS, np.float32)); t.imwrite(d / "z.tif", z)
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "uuid": d.name, "scale": [1.0, 1.0], "voxel_size_um": 8.64}))


def boxes(p):
    lines = p.read_text().splitlines()
    return lines[0], [tuple(map(int, l.split())) for l in lines[1:] if l and not l.startswith("#")]


def ext(*a):
    return [BIN, "ink-ext", "--model", "reader3", "--weights", "none.pth", "--ct-um", "8.64", "--threads", "2", *map(str, a)]


with tempfile.TemporaryDirectory(prefix="ufsm-shell-") as td:
    d = pathlib.Path(td)
    for name, v in (("m", 100), ("s", 200)):
        np.full(SH, v, np.uint8).tofile(d / f"{name}.raw")
        run(BIN, "z3-put", d / name / "v/0", d / f"{name}.raw", "--shape", ",".join(map(str, SH)), "--z0", 0, "--nz", SH[0], "--shard", 128, "--q", 0)
    mesh(d / "A", 100, 600); mesh(d / "B", 100, 1200)
    # --shell-list of A, then B on top of it
    run(*ext("--surface", d / "A", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "s/v/shell.boxes"))
    hdr, ba = boxes(d / "s/v/shell.boxes")
    assert hdr.startswith("# level 0 shard 1024"), hdr
    assert len(ba) == 1 and all(c % 128 == 0 for c in ba[0]), ba
    z, y, x, nz, ny, nx = ba[0]
    assert y <= YS - 12 and y + ny >= YS + 12 and z <= 100 - 10 and 600 + 10 <= z + nz <= 1024 and x <= 20 - 10 and x + nx >= 100 + 10, ba
    run(*ext("--surface", d / "B", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "b.boxes", "--shell-prev", d / "s/v/shell.boxes"))
    _, bb = boxes(d / "b.boxes")
    assert len(bb) == 2 and sorted(b[0] // 1024 for b in bb) == [0, 1], bb
    assert all(b[0] // 1024 == (b[0] + b[3] - 1) // 1024 for b in bb), bb   # each box inside one shard
    redo = (d / "b.boxes.redo").read_text().split()
    assert redo == ["0", "0", "0"], redo   # shard 0 grew; shard 1 is new (nothing to delete)

    def render(shell, tag):
        o = d / f"d{tag}"
        extra = ["--ct-shell", shell] if shell else []
        r = run(*ext("--surface", d / "B", "--ct", d / "m", "v", "--out", d / f"r{tag}", "--render-only", "--dump-layers", o, *extra))
        return np.asarray(t.imread(o / "08.tif")), r.stderr

    L, err = render(d / "s", "a")   # A's boxes: B's far rows outside the shell
    m = re.search(r"DEFAULTED: ink-ext: (\d+) of (\d+) level-0 CT chunks outside the shell", err)
    assert m and 0 < int(m.group(1)) < int(m.group(2)), err[-3000:]
    du = 9.596 / 8.64
    row = lambda yv: int(round((yv - 100) / du))
    assert abs(L.shape[0] - row(1199)) <= 2, L.shape
    inside, outside = L[row(300), 10:60], L[row(1000), 10:60]
    assert (inside == 200).all() and (outside == 100).all(), (inside[:5], outside[:5])
    (d / "s/v/shell.boxes").write_text((d / "b.boxes").read_text())   # B's boxes: everything from the shell
    L, err = render(d / "s", "b")
    assert "DEFAULTED" not in err and (L[row(150):row(1150), 10:60] == 200).all(), (err[-2000:], np.unique(L))
    (d / "e/v").mkdir(parents=True)   # a shell root with no shell.boxes: the q8 store, loudly
    L, err = render(d / "e", "c")
    assert "DEFAULTED: ink-ext: --ct-shell" in err and (L[row(150):row(1150), 10:60] == 100).all(), (err[-2000:], np.unique(L))
    print("test_ink_shell: OK", ba, bb)

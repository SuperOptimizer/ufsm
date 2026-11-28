"""ink-ext --shell-list / --shell-prev / --ct-shell (inkeval 2026-11-27, the q2 CT shell for coarse-scan ink). CPU only: --shell-list and
--render-only load no net and touch no GPU. A two-shard (z) CT whose main store is 100 everywhere and whose shell copy is 200 everywhere;
the shell's coverage is what its shell.boxes says. Checked:
- --shell-list of surface A (z 100..599): x runs of 128^3 chunks (128 x 128 x 128 k boxes, each inside one 1024^3 shard), header
  "# level 0 shard 1024", covering A's rows, all in shard (0, 0, 0); --shell-prev with surface B (z 100..1199): A's chunks kept, shard 0
  grown (named in .redo), shard 1 added (not in .redo);
- the shell quality (header "q Q"): auto = lossless (q 0) while the estimate (chunks x 128^3 x 0.7 B) is <= --shell-max-gb, else q 2
  with "DEFAULTED: ctshell q2 (lossless est N GB > M)"; --shell-q 2 sets it without DEFAULTED; a quality change against --shell-prev
  puts every old shard into .redo (A on A: nothing grew, shard 0 still rewritten);
- latvol --zarr2 1 --volcomp 0 --exact 1 --boxes (a raw zarr v2 source, 1024^3 shards): exactly the listed chunks written (the shard index),
  read back through --ct-shell; --exact 0 (the bounding region per shard) writes more;
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
    assert hdr.startswith("# level 0 shard 1024 q 0 "), hdr
    chunks = lambda bs: {(z // 128, y // 128, (x // 128) + i) for z, y, x, nz, ny, nx in bs for i in range(nx // 128)}
    assert all(c % 128 == 0 for b in ba for c in b) and all(b[3] == 128 and b[4] == 128 and b[0] // 1024 == 0 for b in ba), ba
    assert all(b[2] // 1024 == (b[2] + b[5] - 1) // 1024 for b in ba), ba
    ca = chunks(ba)
    need = {(z // 128, y // 128, x // 128) for z in range(90, 611, 16) for y in (int(YS) - 12, int(YS) + 12) for x in (10, 110)}
    assert need <= ca and len(ca) == len(ba) * 1 + sum(b[5] // 128 - 1 for b in ba), (sorted(need - ca), ba)
    # the quality: an estimate over the limit (A: 10 chunks = 0.015 GB lossless) -> q2, loudly; explicit --shell-q; a change rewrites all
    r = run(*ext("--surface", d / "A", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "q.boxes", "--shell-max-gb", "0.01"))
    m = re.search(r"DEFAULTED: ctshell q2 \(lossless est ([\d.]+) GB > 0.01\)", r.stderr)
    assert m and boxes(d / "q.boxes")[0].startswith("# level 0 shard 1024 q 2 "), r.stderr[-2000:]
    r = run(*ext("--surface", d / "A", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "q.boxes", "--shell-q", "2"))
    assert "DEFAULTED" not in r.stderr and boxes(d / "q.boxes")[0].startswith("# level 0 shard 1024 q 2 "), r.stderr[-2000:]
    r = run(*ext("--surface", d / "A", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "q.boxes", "--shell-prev", d / "s/v/shell.boxes"))
    assert (d / "q.boxes.redo").read_text().split() == [], "same quality, nothing grew: nothing to rewrite"
    r = run(*ext("--surface", d / "A", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "q.boxes", "--shell-prev", d / "s/v/shell.boxes",
                 "--shell-max-gb", "0.01"))
    assert (d / "q.boxes.redo").read_text().split() == ["0", "0", "0"] and "quality differs" in r.stderr, r.stderr[-2000:]
    run(*ext("--surface", d / "B", "--ct", d / "m", "v", "--out", d / "o", "--shell-list", d / "b.boxes", "--shell-prev", d / "s/v/shell.boxes"))
    _, bb = boxes(d / "b.boxes")
    cb = chunks(bb)
    assert ca < cb and {c[0] // 8 for c in cb} == {0, 1}, bb
    assert all(b[3] == 128 and b[4] == 128 and b[2] // 1024 == (b[2] + b[5] - 1) // 1024 for b in bb), bb   # each run inside one shard
    redo = (d / "b.boxes.redo").read_text().split()
    assert redo == ["0", "0", "0"], redo   # shard 0 grew; shard 1 is new (nothing to delete)

    # latvol --exact 1: the listed chunks alone, from a raw zarr v2 source (as from S3), 1024^3 shards
    v2 = d / "v2/v/0"; v2.mkdir(parents=True)
    (v2 / ".zarray").write_text(json.dumps({"shape": list(SH), "chunks": [128, 128, 128], "fill_value": 0, "order": "C", "filters": None,
                                            "dimension_separator": "/", "compressor": None, "zarr_format": 2, "dtype": "|u1"}))
    rng = np.random.default_rng(1); vals = {}
    for cz in range(SH[0] // 128):
        for cy in range(SH[1] // 128):
            c = rng.integers(150, 250, (128, 128, 128), dtype=np.uint8); vals[(cz, cy, 0)] = c
            (v2 / str(cz) / str(cy)).mkdir(parents=True, exist_ok=True); c.tofile(v2 / str(cz) / str(cy) / "0")
    (d / "g/v").mkdir(parents=True)
    (d / "g/v/zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
    lcb = d / "id.lcb"; lcb.write_bytes(np.array([0x424c4655, 1], np.uint32).tobytes() + np.zeros(64 + 64 * 64, np.float32).tobytes())
    def latvol(out, *extra):
        run(BIN, "latvol", lcb, d / "v2", "v", out, "--levels", 0, "--zarr2", 1, "--volcomp", 0, "--gpu", -1, "--threads", 2, "--boxes", d / "c.boxes",
            "--group-from", d / "g", *extra)
        n = 0
        for f in (out / "0/c").rglob("*"):
            if f.is_file():
                ix = np.frombuffer(f.read_bytes()[-(512 * 16 + 4):-4], np.uint64).reshape(-1, 2)
                n += int((ix[:, 0] != np.uint64(2**64 - 1)).sum())
        return n
    # A's chunks plus one at z chunk 7: shard 0's bounding region then spans z chunks 0..7
    (d / "c.boxes").write_text((d / "s/v/shell.boxes").read_text() + "896 128 0 128 128 128\n")
    ne, nb = latvol(d / "x/v", "--exact", 1), latvol(d / "y/v")
    assert (6, 1, 0) not in ca and ne == len(ca | {(7, 1, 0)}) < nb, (ne, len(ca), nb)
    (d / "x/v/shell.boxes").write_text((d / "c.boxes").read_text())

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
    L, err = render(d / "x", "x")   # the latvol --exact copy: A's rows and z chunk 7 from it (source values >= 150), the rest from the main store
    assert "outside the shell" in err and (L[row(300), 10:60] >= 150).all() and (L[row(1000), 10:60] >= 150).all() and (L[row(1150), 10:60] == 100).all() \
        and (L[row(800), 10:60] == 100).all(), (err[-2000:], L[row(300), 10:14], L[row(800), 10:14], L[row(1000), 10:14])
    L, err = render(d / "e", "c")
    assert "DEFAULTED: ink-ext: --ct-shell" in err and (L[row(150):row(1150), 10:60] == 100).all(), (err[-2000:], np.unique(L))
    print("test_ink_shell: OK", len(ca), len(cb), "chunks; latvol exact", ne, "bounding", nb)

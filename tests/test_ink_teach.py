"""ink-teach --extra (B2, 2026-11-18): teacher labels on meshes without human labels. Three flat tifxyz meshes on the 2.399 um grid
(scale 0.05): A (labelled, human stripes), B (an extra mesh 400 voxels away) and C (an extra mesh 10 voxels from A: cleared by
--extra-clear). The teacher maps are the stripe pattern on each mesh's teacher grid (9.596 um px). Checked: A keeps its human-label grid
(label_scale 4 = K x F 1); B gets teacher labels on its own teacher-px raster (label_scale 4 = 9.596 / 2.399, crop = the teacher grid),
the Platt fit of A carried over (ink share of B's labels ~ the stripe share); C is cleared next to A; a duplicate --extra name fails."""
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import tifffile as t

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
G = 60                    # mesh grid points per side (scale 0.05: 20 voxels apart -> 1200 voxels)
LW = 1200                 # human label px per side (F 1: one label px per 2.399 um voxel)
TW = 300                  # teacher px per side (9.596 um = 4 voxels)


def run(*a, ok=True):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    assert (r.returncode == 0) == ok, r.stdout
    return r.stdout


def mesh(d, z0):
    d.mkdir(parents=True)
    v, u = np.mgrid[0:G, 0:G].astype(np.float32) * 20 + 100          # rows = y, cols = x (level-0 voxels)
    t.imwrite(d / "x.tif", u); t.imwrite(d / "y.tif", v); t.imwrite(d / "z.tif", np.full((G, G), z0, np.float32))
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "scale": [0.05, 0.05], "voxel_size_um": 2.399}))


def stripes(n, px_vox):   # ink where (x // 64 voxels) is odd: the same pattern in voxel space on every grid
    c = (np.arange(n) * px_vox) // 64 % 2
    return np.broadcast_to(c[None, :], (n, n)).astype(np.uint8)


with tempfile.TemporaryDirectory(prefix="ufsm-inkteach-") as tmp:
    d = pathlib.Path(tmp)
    mesh(d / "A/tifxyz", 1000.0); mesh(d / "B/tifxyz", 1400.0); mesh(d / "C/tifxyz", 1010.0)
    t.imwrite(d / "A/lab.tif", stripes(LW, 1) * 255); t.imwrite(d / "A/msk.tif", np.full((LW, LW), 255, np.uint8))
    rng = np.random.default_rng(3)
    for n in ("A", "B", "C"):   # the teacher: the stripes as p 0.05 / 0.95 with noise (ink-ext --native-only: grid px (i du, j du))
        p = np.clip(stripes(TW, 4) * 0.9 + 0.05 + rng.normal(0, 0.03, (TW, TW)), 0, 1)
        (d / "maps").mkdir(exist_ok=True); t.imwrite(d / f"maps/{n}.tif", (p * 255 + 0.5).astype(np.uint8))
    segs = {"segments": [{"name": "A", "tifxyz": str(d / "A/tifxyz"), "labels": str(d / "A/lab.tif"), "mask": str(d / "A/msk.tif")}]}
    (d / "segs.json").write_text(json.dumps(segs))
    (d / "extra.tsv").write_text(f"# name tifxyz\nC {d / 'C/tifxyz'}\n")
    out = run(BIN, "ink-teach", "--segs", d / "segs.json", "--teacher", f"t1={d}/maps/%s.tif", "--out", d / "out",
              "--extra", f"B={d / 'B/tifxyz'}", "--extra-list", d / "extra.tsv", "--extra-clear", 48)
    j = json.loads((d / "out/segs-teach.json").read_text())
    S = {e["name"]: e for e in j["segments"]}
    assert set(S) == {"A-teach", "B-teach", "C-teach"}, (S.keys(), out)
    a, b, c = S["A-teach"], S["B-teach"], S["C-teach"]
    assert abs(a["label_scale"] - 4) < 1e-6 and a["crop"][2:] == [LW // 4, LW // 4], a
    assert abs(b["label_scale"] - 9.596 / 2.399) < 1e-6 and b["crop"][2:] == [TW, TW], b
    assert a["teach_px"] == 0, a                                   # A is fully human-supervised: the humans rule, no teacher px
    assert b["teach_px"] > 0.8 * TW * TW, b                        # B: the confident teacher px (the gate drops few: p 0.05 / 0.95)
    share = b["teach_ink_px"] / b["teach_px"]
    assert 0.4 < share < 0.6, (share, b)                           # the stripes are half ink
    assert c["teach_px"] == 0 and "cleared near the labelled meshes" in out, (c, out)
    (d / "dup.tsv").write_text(f"B {d / 'C/tifxyz'}\n")
    run(BIN, "ink-teach", "--segs", d / "segs.json", "--teacher", f"t1={d}/maps/%s.tif", "--out", d / "out2",
        "--extra", f"B={d / 'B/tifxyz'}", "--extra-list", d / "dup.tsv", ok=False)
    # (11-23) a scroll without labelled meshes: --calib-only prints the labelled calibration, --tw + --platt carry it to an empty SEGS;
    # B's labels then equal the run above byte for byte; without --platt / --tw the empty SEGS fails
    cal = run(BIN, "ink-teach", "--segs", d / "segs.json", "--teacher", f"t1={d}/maps/%s.tif", "--out", d / "out3", "--calib-only", 1)
    line = [l for l in cal.splitlines() if l.startswith("tw ")][0].split()
    assert line[2] == "platt" and abs(float(line[1]) - 1) < 1e-9, line
    (d / "empty.json").write_text('{"segments": []}')
    run(BIN, "ink-teach", "--segs", d / "empty.json", "--teacher", f"t1={d}/maps/%s.tif", "--out", d / "out4", "--extra", f"B={d / 'B/tifxyz'}", ok=False)
    run(BIN, "ink-teach", "--segs", d / "empty.json", "--teacher", f"t1={d}/maps/%s.tif", "--out", d / "out4", "--extra", f"B={d / 'B/tifxyz'}",
        "--tw", line[1], "--platt", line[3])
    b4 = {e["name"]: e for e in json.loads((d / "out4/segs-teach.json").read_text())["segments"]}["B-teach"]
    assert abs(b4["teach_px"] - b["teach_px"]) <= 2 and abs(b4["teach_ink_px"] - b["teach_ink_px"]) <= 2, (b4, b)
print(out.strip().splitlines()[-1]); print("test_ink_teach: ok")

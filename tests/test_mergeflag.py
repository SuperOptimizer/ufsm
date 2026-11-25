"""merge-flag (src/mergeflag.c) against a numpy / scipy reference of the same definitions on synthetic raw fields: rings around an
axis whose phase advances once per 4 cells (sp says 4 cells) except in a "merged" half (x >= W / 2) where it advances once per 12 cells
at lower confidence (the density head still says 4). Checks: every cell within 1 byte of the reference, the merged half flagged and the
clean half not, the w = 0 corner 0, a crop without --out refused, and wrap-mf's per-wrap share on a flat tifxyz across both halves."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
D, H, W, h = 40, 72, 80, 2
COEF = (7.5, -4.0, -6.0, -0.2)   # b, c_amp, c_r, c_sp: merged = low confidence, low phase / density turns, narrow spacing
GPU = "--gpu" in sys.argv   # (the suite's gpu line: the GPU path against the CPU's, byte for byte; else every merge-flag on the CPU)
DEV = ["--gpu", "-1"]


def run(*a, ok=True, cpu=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="") if cpu else None   # (count-field: its CPU path, whatever GPU the suite gave)
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120, env=env)
    assert (r.returncode == 0) == ok, r.stdout
    return r.stdout


def reference(pc, ps, sp, w, coef):
    zr = (pc.astype(np.float32) - 128) / 127; zi = (ps.astype(np.float32) - 128) / 127; m = (w > 0).astype(np.float32)
    bf = lambda a, s: ndi.uniform_filter(a.astype(np.float64), s, mode='nearest')
    amp9 = bf(np.hypot(zr, zi) * m, 9) / np.maximum(bf(m, 9), 1e-3)
    spc = sp.astype(np.float32) / 8; known = (sp > 0).astype(np.float32)
    sp9 = bf(spc, 9) / np.maximum(bf(known, 9), 1e-3); sp5 = bf(spc, 5) / np.maximum(bf(known, 5), 1e-3)
    ka = np.zeros(zr.shape)
    for a in range(3):
        n = zr.shape[a]; i0 = np.minimum(np.arange(n), n - 2); i1 = i0 + 1
        ar, ai = np.take(zr, i0, axis=a), np.take(zi, i0, axis=a); br, bi = np.take(zr, i1, axis=a), np.take(zi, i1, axis=a)
        cr = bf(br * ar + bi * ai, 5); ci = bf(bi * ar - br * ai, 5)
        ka += (np.arctan2(ci, cr) / (2 * np.pi)) ** 2
    r5 = np.sqrt(ka) * sp5
    lg = coef[0] + coef[1] * amp9 + coef[2] * r5 + coef[3] * sp9
    return np.where(w > 0, np.rint(255 / (1 + np.exp(-lg))), 0).astype(np.uint8)


with tempfile.TemporaryDirectory(prefix="ufsm-mflag-") as tmp:
    t = pathlib.Path(tmp); f = t / "fields"; f.mkdir()
    cy, cx = -30.0, W / 2   # axis outside the box (cells): rings are nearly straight across x
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    r = np.hypot(y + 0.5 - cy, x + 0.5 - cx)
    merged = x >= W // 2
    ph = np.where(merged, r / 12.0, r / 4.0); conf = np.where(merged, 0.35, 0.85)
    rng = np.random.default_rng(7)
    pc = np.clip(np.rint(128 + 127 * conf * np.cos(2 * np.pi * ph) + rng.normal(0, 3, ph.shape)), 1, 255).astype(np.uint8)
    ps = np.clip(np.rint(128 + 127 * conf * np.sin(2 * np.pi * ph) + rng.normal(0, 3, ph.shape)), 1, 255).astype(np.uint8)
    sp = np.full(ph.shape, 32, np.uint8)   # 4 cells
    w = np.full(ph.shape, 255, np.uint8); w[:6, :6, :6] = 0
    tn = np.zeros(ph.shape, np.uint8)
    for k, a in (("pc", pc), ("ps", ps), ("sp", sp), ("w", w), ("t0", tn), ("t1", tn + 128), ("t2", tn + 128), ("t3", tn + 128), ("t4", tn + 128), ("t5", tn + 128)):
        a.tofile(f / f"{k}.raw")
    axis = [[cy * h, cx * h]] * (D * h + 1)
    (f / "fields.json").write_text(json.dumps({"format": "raw", "origin": [0, 0, 0], "shape": [D, H, W], "h": h, "um": 9.6, "axis": axis}))
    out = run(BIN, "merge-flag", f, "--coef", ",".join(map(str, COEF)), "--thr", "128", "--threads", "3", *DEV)
    print(out.strip())
    mf = np.fromfile(f / "mf.raw", np.uint8).reshape(D, H, W)
    ref = reference(pc, ps, sp, w, COEF)
    d = np.abs(mf.astype(int) - ref.astype(int))
    print(f"merge-flag vs reference: max |diff| {d.max()}, cells off by 1: {np.mean(d == 1):.5f}")
    assert d.max() <= 1, np.argwhere(d > 1)[:10]
    assert np.mean(d == 1) < 0.01
    inner = (slice(8, -8), slice(8, -8))
    mm = mf[:, :, W // 2 + 6:][:, 8:-8].mean() / 255; mc = mf[:, :, :W // 2 - 6][:, 8:-8][6:].mean() / 255
    print(f"mean mf: merged half {mm:.3f}, clean half {mc:.3f}")
    assert mm > 0.8 and mc < 0.2, (mm, mc)
    assert (mf[:6, :6, :6] == 0).all()
    # a crop (raw_offset) without --out is refused; with --out it is the same cells
    c = t / "crop"; c.mkdir()
    for k in ("pc", "ps", "sp", "w", "t0", "t1", "t2", "t3", "t4", "t5"): (c / f"{k}.raw").symlink_to(f / f"{k}.raw")
    (c / "fields.json").write_text(json.dumps({"format": "raw", "origin": [0, 20, 20], "shape": [D, H - 10, W - 10], "h": h, "um": 9.6,
                                               "raw_offset": [0, 10, 10], "raw_shape": [D, H, W], "axis": [[cy * h - 20, cx * h - 20]] * (D * h + 1)}))
    run(BIN, "merge-flag", c, "--coef", ",".join(map(str, COEF)), *DEV, ok=False)
    run(BIN, "merge-flag", c, "--coef", ",".join(map(str, COEF)), "--out", t / "cmf", *DEV)
    cm = np.fromfile(t / "cmf" / "mf.raw", np.uint8).reshape(D, H - 10, W - 10)
    assert np.abs(cm[:, 6:, 6:].astype(int) - mf[:, 16:, 16:].astype(int)).max() <= 1   # (away from the crop's own replicated border)
    # wrap-mf: a flat wrap at z = 40 native across the box (x native 8 .. 152): half its vertices in the merged half
    wd = t / "wraps" / "w005"; wd.mkdir(parents=True)
    gy, gx = np.meshgrid(np.arange(20, 120, 4, dtype=np.float32), np.arange(8, 152, 4, dtype=np.float32), indexing='ij')
    from struct import pack

    def tif(path, a):   # minimal float32 TIFF (one strip) for the tifxyz grids
        a = np.ascontiguousarray(a, '<f4'); hh, ww = a.shape; data = a.tobytes(); n = 9
        ifd = pack('<H', n) + b''.join(pack('<HHII', tag, typ, cnt, val) for tag, typ, cnt, val in
                                         [(256, 4, 1, ww), (257, 4, 1, hh), (258, 3, 1, 32), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8 + 2 + n * 12 + 4),
                                          (277, 3, 1, 1), (279, 4, 1, len(data)), (339, 3, 1, 3)]) + pack('<I', 0)
        path.write_bytes(b'II*\x00' + pack('<I', 8) + ifd + data)
    tif(wd / "x.tif", gx); tif(wd / "y.tif", gy); tif(wd / "z.tif", np.full(gx.shape, 40, np.float32))
    (wd / "meta.json").write_text(json.dumps({"type": "seg", "format": "tifxyz", "scale": [0.25, 0.25], "voxel_size_um": 9.6, "winding": 5}))
    out = run(BIN, "wrap-mf", t / "wraps", "--fields", f, "--thr", "128", "--pool", "1", "--out", t / "mfw", "--tsv", t / "mf.tsv")
    print(out.strip())
    row = (t / "mf.tsv").read_text().splitlines()[1].split('\t')
    exp = np.mean(mf[20 // h, (gy // h).astype(int), (gx // h).astype(int)] >= 128)
    assert row[0] == "w005" and abs(float(row[2]) - exp) < 1e-3, (row, exp)
    assert 0.4 < exp < 0.6, exp
    from PIL import Image
    g = np.array(Image.open(t / "mfw" / "w005_mf.tif")); assert g.shape == gx.shape, g.shape   # the vertex grid
    run(BIN, "wrap-mf", t / "wraps", "--fields", f, "--thr", "128", "--pool", "1", "--out", t / "mfr", "--render", "1")
    rr = np.array(Image.open(t / "mfr" / "w005_mf.tif"))
    assert rr.shape == ((gx.shape[0] - 1) * 4 + 1, (gx.shape[1] - 1) * 4 + 1), rr.shape   # ufsm render --scale 1's pixel grid (step 4)
    assert (rr[::4, ::4] == g).all()
    # merge-flag --eval on the same fields against synthetic GT codes: true sheets every 8 native voxels on both halves (the merged half's
    # phase runs at a third of their rate), codes via ingest-labels (zarr v2 -> the volcomp store the eval reads)
    zn, yn, xn = D * h, H * h, W * h
    Z, Y, X = np.meshgrid(np.arange(zn), np.arange(yn), np.arange(xn), indexing='ij')
    rr = np.hypot(Y + 0.5 - cy * h, X + 0.5 - cx * h); k = np.rint(rr / 8)
    codes = np.where(np.abs(rr - 8 * k) < 0.6, 1 + 18 * (k.astype(int) % 14), 0).astype(np.uint8)
    lv = t / "gt.zarr" / "0"; lv.mkdir(parents=True); C = 16   # (16^3 chunks: the 128 shard must be a multiple of them)
    (lv / ".zarray").write_text(json.dumps({"shape": [zn, yn, xn], "chunks": [C] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                            "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
    for a in range(zn // C):
        for b in range(yn // C):
            (lv / str(a) / str(b)).mkdir(parents=True, exist_ok=True)
            for cc in range(xn // C): (lv / str(a) / str(b) / str(cc)).write_bytes(codes[a * C:(a + 1) * C, b * C:(b + 1) * C, cc * C:(cc + 1) * C].tobytes())
    (t / "lut.txt").write_text(" ".join(str(min(v, 254)) for v in range(256)))
    run(BIN, "ingest-labels", t, "gt.zarr", t / "codes", "--um", "9.6", "--lut", t / "lut.txt", "--levels", "1", "--shard", "128", "--threads", "2")
    (t / "ev.tsv").write_text(f"a-B1 paris4 {f} {t / 'codes'} 9.6\nb-B2 1667 {f} {t / 'codes'} 9.6\n")
    r = subprocess.run([str(BIN), "merge-flag", "--eval", str(t / "ev.tsv"), "--threads", "2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    print(r.stdout.strip()); print(r.stderr.strip())
    assert r.returncode == 0, r.stderr
    rule = [l for l in r.stdout.splitlines() if "the rule" in l]
    assert len(rule) >= 3, r.stdout
    for l in rule:
        auc = float(l.split(" AUC ")[1].split()[0]); mer = float(l.split(" merged ")[1].split()[0])
        assert auc > 0.95 and 0.3 < mer < 0.7, l   # (about half the intervals lie in the merged half)
    # merge-flag --sheet-pairs on the same GT rings (sheets every 8 native voxels = 76.8 um): every triplet's s lies on a's sheet, b on the
    # adjacent one (radius from the axis), gaps ~76.8 um; the clean half's phase (try9 column) picks the right sheet, a, s matched in distance
    run(BIN, "merge-flag", "--sheet-pairs", t / "ev.tsv", "--out", t / "sp", "--max-per-box", "400")
    rows = [l.split('\t') for l in (t / "sp.tsv").read_text().splitlines() if l and not l.startswith(("box", "#", "name"))]
    assert len(rows) > 100, len(rows)
    P = {s: np.fromfile(t / f"sp.{s}.pts", np.float32).reshape(-1, 3, 3) for s in ("paris4", "1667")}
    for s, p in P.items():
        assert len(p) == sum(r[1] == s for r in rows), (s, len(p))
        rad = np.hypot(p[..., 1] + 0.5 - cy * h, p[..., 2] + 0.5 - cx * h)   # (absolute native = the box's: origin 0)
        sh = np.rint(rad / 8)
        assert (sh[:, 0] == sh[:, 1]).all() and (np.abs(sh[:, 2] - sh[:, 0]) == 1).all(), np.argwhere(sh[:, 0] != sh[:, 1])[:5]
    gap = np.array([float(r[2]) for r in rows]); dsm = np.array([float(r[3]) for r in rows]); dad = np.array([float(r[4]) for r in rows])
    assert np.all(np.abs(gap - 76.8) <= 2 * 9.6), (gap.min(), gap.max())
    assert np.all(np.abs(dsm - dad) <= np.maximum(0.5 * gap, 1)), np.abs(dsm - dad).max()
    clean = np.array([float(r[11]) < W * h / 2 - 12 and float(r[13]) < W * h / 2 - 12 for r in rows])   # (s and b in the clean half)
    t9s = np.array([float(r[5]) for r in rows]); t9a = np.array([float(r[6]) for r in rows])
    acc = np.mean(t9s[clean] < t9a[clean]); print(f"sheet-pairs: {len(rows)} triplets, gap {gap.mean():.1f} um, clean-half phase picks {acc:.3f}")
    assert clean.sum() > 20 and acc > 0.95, (clean.sum(), acc)
    # --dense-pts / --dense-cos plumbing: probe pairs per box, then dense features (a .dvf as dinovol-probe pfeats --dense-only writes it)
    # that say "same sheet" (cos 1) on the merged half's intervals and "other sheet" (cos 0) on the clean half's: dcos alone is then perfect
    run(BIN, "merge-flag", "--eval", t / "ev.tsv", "--threads", "2", "--dense-pts", t)
    for nm in ("a-B1", "b-B2"):
        q = np.fromfile(t / f"{nm}.pts", np.float32).reshape(-1, 2, 3); n = len(q); mx = q[:, :, 2].mean(1)
        F = 4; X = np.zeros((n, 2, F), np.float16); X[:, 0, 0] = 1; X[:, 1, 0] = np.where(mx >= W * h / 2, 1, 0); X[:, 1, 1] = np.where(mx >= W * h / 2, 0, 1)
        hdr = b"DVPF0001" + np.array([F, 0, F, 0, 1, 0, 0, 0], np.int32).tobytes() + b"\0" * 256
        rec = np.zeros(2 * n, dtype=[("r", "V16"), ("x", "<f2", (F,))]); rec["x"] = X.reshape(2 * n, F)
        (t / f"{nm}.dvf").write_bytes(hdr + rec.tobytes())
    r = subprocess.run([str(BIN), "merge-flag", "--eval", str(t / "ev.tsv"), "--threads", "2", "--dense-cos", str(t)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    dl = [l for l in r.stdout.splitlines() if "all test boxes" in l and "dcos alone" in l]; print("\n".join(dl))
    assert len(dl) == 3 and all(float(l.split(" AUC ")[1].split()[0]) > 0.95 for l in dl), r.stdout
    assert "c_dcos" in r.stdout and "by GT gap" in r.stdout
    # zarr fields (each plane ingested: values clipped at 254): merge-flag --weighted DIR (the unroll's MFREF ref input) equals
    # merge-flag --apply on the same fields and mf; pc ps scaled toward 128 by 1 - mf / 255, w by it (>= 1 on material), others linked
    zf = t / "zfields"; zf.mkdir()
    for k, a in (("pc", pc), ("ps", ps), ("sp", sp), ("w", w)):
        lk = t / f"{k}.zarr" / "0"; lk.mkdir(parents=True)
        (lk / ".zarray").write_text(json.dumps({"shape": [D, H, W], "chunks": [8] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                                "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
        for a0 in range(D // 8):
            for b0 in range(H // 8):
                (lk / str(a0) / str(b0)).mkdir(parents=True, exist_ok=True)
                for c0 in range(W // 8): (lk / str(a0) / str(b0) / str(c0)).write_bytes(a[a0 * 8:a0 * 8 + 8, b0 * 8:b0 * 8 + 8, c0 * 8:c0 * 8 + 8].tobytes())
        run(BIN, "ingest-labels", t, f"{k}.zarr", t / f"z{k}", "--um", "9.6", "--lut", t / "lut.txt", "--levels", "1", "--shard", "128", "--threads", "2")
        (zf / k).symlink_to(t / f"z{k}" / "9.6")
    for k in range(6): (zf / f"t{k}").symlink_to(t / "zsp" / "9.6")   # (the normal planes count-field reads; linked as is by --weighted)
    (zf / "fields.json").write_text(json.dumps({"format": "zarr", "origin": [0, 0, 0], "shape": [D, H, W], "h": h, "um": 9.6, "axis": axis}))
    run(BIN, "merge-flag", zf, "--coef", ",".join(map(str, COEF)), "--weighted", t / "wtd", "--threads", "3", *DEV)
    run(BIN, "merge-flag", "--apply", zf, t / "app", "--threads", "3")
    rd = lambda root, k: (run(BIN, "read", root, k, 0, 0, 0, D, H, W, t / "r.raw"), np.fromfile(t / "r.raw", np.uint8).reshape(D, H, W))[1]
    mz = rd(zf, "mf"); wz = rd(zf, "w")
    for k in ("pc", "ps", "w"):
        a, b = rd(t / "wtd", k), rd(t / "app", k)
        assert (a == b).all(), k
    f_ = np.maximum(0, 1 - mz / 255.0)
    pz = rd(zf, "pc"); exp = np.where(wz > 0, np.rint(128 + (pz.astype(float) - 128) * f_), pz)
    assert np.abs(rd(t / "wtd", "pc") - exp).max() <= 1
    ww = rd(t / "wtd", "w"); assert (ww[wz > 0] >= 1).all() and (ww[mz > 200] < wz[mz > 200]).all()
    assert (t / "wtd" / "t0").is_symlink() and (t / "wtd" / "sp").is_symlink()
    # count-field --mf-w 1 (the unroll's MFREF: the weighting inside every field load) == count-field on merge-flag --apply's planes
    cfa = ["--R", "0", "--pool", "4", "--model-only", "1", "--model-stride", "1", "--dir", "1", "--iters", "50", "--threads", "2"]
    run(BIN, "count-field", t / "app", t / "cfa.unused", *cfa, "--model", t / "ma", cpu=True)
    run(BIN, "count-field", zf, t / "cfb.unused", *cfa, "--model", t / "mb", "--mf-w", "1", cpu=True)
    fa = sorted(p.name for p in (t / "ma").iterdir()); assert fa == sorted(p.name for p in (t / "mb").iterdir()), fa
    for n in fa: assert (t / "ma" / n).read_bytes() == (t / "mb" / n).read_bytes(), n
    run(BIN, "count-field", f, t / "cfc.unused", *cfa, "--model", t / "mc", "--mf-w", "1", ok=False, cpu=True)   # (raw fields without an mf plane: refused)
    if GPU:   # the GPU path (src/nn_mf.cu) byte for byte against the CPU's: raw fields, a crop, zarr fields (2 x 5 tiles in z, y: --gpu-mem 0.0035)
        for src, extra in ((f, []), (c, ["--out", t / "cg"]), (zf, ["--out", t / "zg"])):
            g = run(BIN, "merge-flag", src, "--coef", ",".join(map(str, COEF)), "--thr", "128", "--gpu", "0", "--gpu-mem", "0.0035", *extra)
            assert "DEFAULTED" not in g, g
        cpu = lambda d, k: np.fromfile(d / k, np.uint8)
        assert (np.fromfile(f / "mf.raw", np.uint8) == mf.ravel()).all()   # (the raw run above overwrote f/mf.raw: the same bytes)
        assert (cpu(t / "cg", "mf.raw") == cpu(t / "cmf", "mf.raw")).all()
        assert (rd(t / "zg", "mf") == mz).all()
    print("test_mergeflag: PASS")

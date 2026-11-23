"""ink-place / obj-tifxyz / the TIFF predictor-2 float read (hfink 2026-11-23).
ink-place: a 2-D crop (0 background / 1 ink / else unknown) placed on a flat tifxyz canvas at a fractional corner, anisotropic scale and an
orientation; the labels / mask zarrs byte for byte against a numpy reference of its rule (bilinear over the crop's px, supervised only when
every weighted neighbour is known, ink where the share >= 0.5), label_scale F = floor(px_um / voxel um), the segs.json window.
obj-tifxyz: a planar OBJ (uv -> render px, v flipped) -> grid points = the plane's affine map (exact up to float32).
The tifxyz is written with predictor 2 (horizontal differencing on 32-bit floats: the open-data tifxyz_original grids) where tifffile can."""
import json
import pathlib
import subprocess
import sys
import tempfile

import importlib.util

_miss = [m for m in ("numpy", "tifffile", "zarr") if importlib.util.find_spec(m) is None]
if _miss:
    print(f"SKIP: no python {', '.join(_miss)} (the reference needs it)")
    sys.exit(0)

import numpy as np
import tifffile as t
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout
    return r.stdout


def write_tifxyz(d, X, Y, Z, scale, vum, pred2):
    d.mkdir(parents=True, exist_ok=True)
    used = False
    for n, a in (("x", X), ("y", Y), ("z", Z)):
        a = np.ascontiguousarray(a, np.float32)
        if pred2:
            try:   # predictor 2 on float32 (libtiff adds the 32-bit words as integers)
                t.imwrite(d / f"{n}.tif", a, compression="zlib", predictor=2)
                used = True
                continue
            except Exception:
                pass
        t.imwrite(d / f"{n}.tif", a)
    (d / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "uuid": d.name, "scale": [scale, scale], "voxel_size_um": vum}))
    return used


def write_pred2_f32(path, a):
    """a minimal uncompressed float32 TIFF with predictor 2: each row's 32-bit words differenced as unsigned integers"""
    import struct
    a = np.ascontiguousarray(a, np.float32)
    h, w = a.shape
    u = a.view(np.uint32).astype(np.uint64)
    d = u.copy()
    d[:, 1:] = (u[:, 1:] - u[:, :-1]) % (1 << 32)
    data = d.astype(np.uint32).tobytes()
    tags = [(256, 4, w), (257, 4, h), (258, 3, 32), (259, 3, 1), (262, 3, 1), (273, 4, 8), (277, 3, 1), (278, 4, h), (279, 4, len(data)), (317, 3, 2), (339, 3, 3)]
    ifd_off = 8 + len(data)
    b = b"II" + struct.pack("<HI", 42, ifd_off) + data + struct.pack("<H", len(tags))
    for tag, typ, val in tags:
        b += struct.pack("<HHI", tag, typ, 1) + (struct.pack("<HH", val, 0) if typ == 3 else struct.pack("<I", val))
    b += struct.pack("<I", 0)
    pathlib.Path(path).write_bytes(b)


def orient(c, o):
    if o & 1:
        c = c.T
    if o & 2:
        c = c[::-1]
    if o & 4:
        c = c[:, ::-1]
    return c


def reference(crop, o, y0, x0, sy, sx, F, RH, RW):
    c = orient(crop, o).astype(np.int64)
    h, w = c.shape
    r0, r1 = int(np.floor(y0 / F)), int(np.ceil((y0 + h * sy) / F))
    q0, q1 = int(np.floor(x0 / F)), int(np.ceil((x0 + w * sx) / F))
    r0, q0, r1, q1 = max(r0, 0), max(q0, 0), min(r1, RH), min(q1, RW)
    L = np.zeros((RH, RW), np.uint8)
    M = np.zeros((RH, RW), np.uint8)
    for R in range(r0, r1):
        u = ((R + 0.5) * F - y0) / sy - 0.5
        yy0 = int(np.floor(u))
        fy = u - yy0
        for Q in range(q0, q1):
            v = ((Q + 0.5) * F - x0) / sx - 0.5
            xx0 = int(np.floor(v))
            fx = v - xx0
            s = sw = 0.0
            bad = False
            for dy in (0, 1):
                for dx in (0, 1):
                    g = (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                    if g <= 0:
                        continue
                    a, b = yy0 + dy, xx0 + dx
                    if a < 0 or a >= h or b < 0 or b >= w or c[a, b] > 1:
                        bad = True
                        break
                    s += g * c[a, b]
                    sw += g
                if bad:
                    break
            if bad or sw <= 0:
                continue
            M[R, Q] = 255
            L[R, Q] = 255 if s / sw >= 0.5 else 0
    return L, M, (r0, q0, r1, q1)


rng = np.random.default_rng(7)
with tempfile.TemporaryDirectory(prefix="ufsm-inkplace-") as tmp:
    d = pathlib.Path(tmp)
    # a flat 2.4 um mesh: grid 40 x 30 vertices, step 20 -> canvas 600 x 800 (rows x cols); point = (z 100 + row, y 7 col, x 5 row)
    gh, gw, st = 30, 40, 20
    rr, cc = np.meshgrid(np.arange(gh, dtype=np.float64), np.arange(gw, dtype=np.float64), indexing="ij")
    Z, Y, X = 100 + rr * st, 7 * cc * st / st, 5 * rr * st / st
    Z[0, 0] = -1   # an invalid vertex (outside the papyrus): does not affect the label raster
    pred2 = write_tifxyz(d / "mesh", X, Y, Z, 1 / st, 2.4, True)
    # crop: blobs of ink, background, ignore (2) and outside (255)
    crop = np.zeros((47, 61), np.uint8)
    crop[rng.random(crop.shape) < 0.3] = 1
    crop[5:12, 8:20] = 2
    crop[30:, 50:] = 255
    t.imwrite(d / "crop.tif", crop)
    cases = [(0, 37.3, 51.7, 3.9, 3.7), (2, 101.6, 9.25, 4.1, 3.3), (5, 0.4, 250.9, 3.6, 4.2)]
    lines = []
    for i, (o, y0, x0, sy, sx) in enumerate(cases):
        lines.append(f"seg{i}\t{d / 'mesh'}\t{d / 'crop.tif'}\t{y0}\t{x0}\t{sy}\t{sx}\t-\t-\tcase{i}\t{o}\n")
    (d / "place.tsv").write_text("".join(lines))
    out = run(BIN, "ink-place", d / "place.tsv", d / "out", "--px-um", 4.8)
    j = json.loads((d / "out" / "segs.json").read_text())["segments"]
    assert len(j) == len(cases), j
    F = 2   # floor(4.8 / 2.4)
    RH, RW = int(np.ceil(gh * st / F)), int(np.ceil(gw * st / F))
    for i, (o, y0, x0, sy, sx) in enumerate(cases):
        Lr, Mr, (r0, q0, r1, q1) = reference(crop, o, y0, x0, sy, sx, F, RH, RW)
        e = j[i]
        assert e["label_scale"] == F and e["crop"] == [q0, r0, q1 - q0, r1 - r0], (e, (r0, q0, r1, q1))
        L = np.asarray(zarr.open(e["labels"], mode="r")[:])
        M = np.asarray(zarr.open(e["mask"], mode="r")[:])
        assert L.shape == (RH, RW) and M.shape == (RH, RW), (L.shape, RH, RW)
        dl, dm = int((L != Lr).sum()), int((M != Mr).sum())
        assert dl == 0 and dm == 0, f"case {i}: {dl} label / {dm} mask px differ (of {int((Mr > 0).sum())} supervised)"
        assert (Mr > 0).sum() > 1000 and (Lr > 0).sum() > 100
    # --skip-ref-sup with the placed labels themselves as REF: everything is left out; the REF check sees r = 1 at offset 0
    e = j[0]
    (d / "place2.tsv").write_text(lines[0].replace("\t-\t-\t", f"\t{e['labels']}\t{e['mask']}\t"))
    out2 = run(BIN, "ink-place", d / "place2.tsv", d / "out2", "--px-um", 4.8, "--search", 3, "--skip-ref-sup")
    row = [ln.split("\t") for ln in out2.splitlines() if ln.startswith("seg0")][0]
    assert float(row[3]) == 0 and abs(float(row[7]) - 1) < 1e-6 and row[8] == "0" and row[9] == "0", row

    # obj-tifxyz: a planar quad, uv over [0.1, 0.9]^2 of a 400 x 300 render; the point at render px (r, c) is affine in (r, c)
    W, H = 400, 300
    A = np.array([[0.5, 2.0, 10.0], [1.5, -0.25, 20.0], [0.0, 0.75, 30.0]])   # (x, y, z) = A @ (col, row, 1)
    corners = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    obj = []
    for u, v in corners:
        col, row = u * W, (1 - v) * H
        x, y, z = A @ np.array([col, row, 1.0])
        obj.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    for u, v in corners:
        obj.append(f"vt {u} {v}\n")
    obj.append("f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    (d / "q.obj").write_text("".join(obj))
    run(BIN, "obj-tifxyz", d / "q.obj", d / "qt", "--img", f"{W},{H}", "--step", 10, "--flip-v", 1, "--vox-um", 7.91)
    gx, gy, gz = (t.imread(d / "qt" / f"{n}.tif") for n in "xyz")
    meta = json.loads((d / "qt" / "meta.json").read_text())
    assert abs(meta["scale"][0] - 0.1) < 1e-9 and gx.shape == (30, 40), (meta, gx.shape)
    ii, jj = np.meshgrid(np.arange(30) * 10.0, np.arange(40) * 10.0, indexing="ij")
    inside = (jj >= 0.1 * W + 1e-6) & (jj <= 0.9 * W - 1e-6) & (ii >= 0.1 * H + 1e-6) & (ii <= 0.9 * H - 1e-6)
    want = [A[k, 0] * jj + A[k, 1] * ii + A[k, 2] for k in range(3)]
    for g, w in zip((gx, gy, gz), want):
        assert np.all(g[inside] >= -0.5) and np.abs(g[inside] - w[inside]).max() < 1e-2, np.abs(g[inside] - w[inside]).max()
    outside = (jj < 0.1 * W - 1e-6) | (jj > 0.9 * W + 1e-6) | (ii < 0.1 * H - 1e-6) | (ii > 0.9 * H + 1e-6)
    assert np.all(gz[outside] == -1)

    # predictor 2 on float32 (the open-data tifxyz_original grids): tifxyz-scale reads it (img_read_f32) and writes it back plain
    (d / "p2s" / "p2").mkdir(parents=True)
    vals = (np.abs(rng.standard_normal((13, 29))) * 1000 + 10).astype(np.float32)   # (z < 0 marks an invalid vertex)
    vals[3, 4] = -1
    for n in "xyz":
        write_pred2_f32(d / "p2s" / "p2" / f"{n}.tif", vals + {"x": 0, "y": 1, "z": 2}[n])
    (d / "p2s" / "p2" / "meta.json").write_text(json.dumps({"format": "tifxyz", "type": "seg", "uuid": "p2", "scale": [0.05, 0.05]}))
    run(BIN, "tifxyz-scale", d / "p2s", d / "p2b", 1)   # (a directory of wraps)
    for n in "xyz":
        got = t.imread(d / "p2b" / "p2" / f"{n}.tif")
        want = vals + {"x": 0, "y": 1, "z": 2}[n]
        assert got.shape == want.shape and np.array_equal(got[vals != -1], want[vals != -1]), (n, np.abs(got - want).max())
print("ok: ink-place 3 placements byte-exact vs numpy, REF / skip-ref-sup, obj-tifxyz planar exact, predictor-2 float32 TIFF read")

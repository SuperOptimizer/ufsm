"""ct-resample: a CT level trilinear to another voxel size over boxes (8.64 -> 4.8 um, f = 0.5556, centres aligned, edges clamped,
floor(v + 0.5)) against a numpy reference inside both boxes (|diff| <= 1 at rounding ties only, <= 0.01 % of voxels), absent outside the
boxes' shards (0), and the output group's OME scale 4.8."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

if importlib.util.find_spec("numpy") is None:
    print("SKIP: no python numpy (the reference needs it)")
    sys.exit(0)
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("UFSM_BIN", ROOT / "build/ufsm"))
U0, U = 8.64, 4.8


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    assert r.returncode == 0, r.stdout
    return r.stdout


def read(root, key, o, n, td):
    p = pathlib.Path(td) / "rd.raw"
    run(BIN, "read", root, key, *o, *n, p, "--threads", 4)
    return np.fromfile(p, np.uint8).reshape(n)


def taps(lo, hi, f, n):
    s = (np.arange(lo, hi) + 0.5) * f - 0.5
    i0 = np.floor(s).astype(np.int64); w = s - i0; i1 = i0 + 1
    low = s <= 0; i0[low] = 0; i1[low] = 0; w[low] = 0
    high = i0 >= n - 1; i0[high] = n - 1; i1[high] = n - 1; w[high] = 0
    return i0, i1, w


def ref(V, box_o, f):
    (z0, y0, x0), (z1, y1, x1) = box_o[:3], box_o[3:]
    tz, ty, tx = taps(z0, z1, f, V.shape[0]), taps(y0, y1, f, V.shape[1]), taps(x0, x1, f, V.shape[2])
    V = V.astype(np.float64)
    def lx(A): return A[..., tx[0]] + tx[2] * (A[..., tx[1]] - A[..., tx[0]])
    def ly(A): return A[:, ty[0]] + ty[2][None, :, None] * (A[:, ty[1]] - A[:, ty[0]])
    A = ly(lx(V))
    A = A[tz[0]] + tz[2][:, None, None] * (A[tz[1]] - A[tz[0]])
    return np.floor(A + 0.5).astype(np.uint8)


def cref(V, box_o, f):   # ct-resample --mode cubic: Catmull-Rom (a = -0.5), taps clamped to the volume, x then y then z, clamp + round
    (z0, y0, x0), (z1, y1, x1) = box_o[:3], box_o[3:]
    def ct(lo, hi, n):
        s = (np.arange(lo, hi) + 0.5) * f - 0.5; fl = np.floor(s); t = s - fl; a = -0.5; t2 = t * t; t3 = t2 * t
        w = [a * (t3 - 2 * t2 + t), (a + 2) * t3 - (a + 3) * t2 + 1, -(a + 2) * t3 + (2 * a + 3) * t2 - a * t, a * (t3 - t2)]
        i = [np.clip(fl.astype(np.int64) - 1 + k, 0, n - 1) for k in range(4)]
        return i, w
    (iz, wz), (iy, wy), (ix, wx) = ct(z0, z1, V.shape[0]), ct(y0, y1, V.shape[1]), ct(x0, x1, V.shape[2])
    V = V.astype(np.float64)
    A = wx[0] * V[..., ix[0]] + wx[1] * V[..., ix[1]] + wx[2] * V[..., ix[2]] + wx[3] * V[..., ix[3]]
    A = wy[0][:, None] * A[:, iy[0]] + wy[1][:, None] * A[:, iy[1]] + wy[2][:, None] * A[:, iy[2]] + wy[3][:, None] * A[:, iy[3]]
    A = wz[0][:, None, None] * A[iz[0]] + wz[1][:, None, None] * A[iz[1]] + wz[2][:, None, None] * A[iz[2]] + wz[3][:, None, None] * A[iz[3]]
    return np.where(A <= 0, 0, np.where(A >= 255, 255, np.floor(A + 0.5))).astype(np.uint8)


def main():
    rng = np.random.default_rng(7)
    sh = (200, 160, 170)
    g = np.indices(sh).astype(np.float64)   # smooth structure + noise (interpolation errors show on both)
    V = np.clip(110 + 60 * np.sin(g[0] / 7.3) * np.cos(g[1] / 5.1) + 30 * np.sin(g[2] / 3.7) + rng.normal(0, 12, sh), 0, 255).astype(np.uint8)
    V[:, :, :3] = 0
    with tempfile.TemporaryDirectory(prefix="ufsm-ctres-") as td:
        t = pathlib.Path(td); V.tofile(t / "in.raw")
        run(BIN, "z3-put", t / "s/v/0", t / "in.raw", "--shape", ",".join(map(str, sh)), "--z0", 0, "--nz", sh[0], "--shard", 128, "--q", 0)
        Vr = read(t / "s", "v/0", (0, 0, 0), sh, td)
        assert np.array_equal(Vr, V), "z3-put --q 0 not lossless"
        boxes = [(0, 0, 0, 40, 50, 60), (130, 90, 100, 70, 70, 70)]   # (one at the volume's low corner, one at its high corner: both clamps)
        out = run(BIN, "ct-resample", t / "s", "v", t / "o/r", "--um", U0, "--to", U, "--level", 0, "--shard", 128, "--threads", 3,
                  *sum((["--box", ",".join(map(str, b))] for b in boxes), []))
        print(out.strip())
        f = U / U0
        osh = [int(np.ceil(n / f - 1e-9)) for n in sh]
        meta = json.loads((t / "o/r/4.8/zarr.json").read_text())
        assert meta["shape"] == osh, (meta["shape"], osh)
        grp = json.loads((t / "o/r/zarr.json").read_text())
        assert grp["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][0] == U
        bad = tot = 0
        for b in boxes:
            lo = [int(np.floor(b[d] / f + 1e-9)) for d in range(3)]
            hi = [min(osh[d], int(np.ceil((b[d] + b[3 + d]) / f - 1e-9))) for d in range(3)]
            R = ref(V, (*lo, *hi), f)
            C = read(t / "o", "r/4.8", lo, [hi[d] - lo[d] for d in range(3)], td)
            dif = np.abs(C.astype(int) - R.astype(int))
            assert dif.max() <= 1, f"box {b}: max |diff| {dif.max()}"
            bad += int((dif > 0).sum()); tot += dif.size
        assert bad <= 1e-4 * tot, f"{bad} of {tot} voxels differ by 1"
        far = read(t / "o", "r/4.8", (0, osh[1] - 40, 0), (40, 40, 40), td)   # (a shard under no box: absent, 0)
        assert not far.any(), "a shard outside the boxes was written"
        # cubic (2026-11-25): an integer factor (2: 8.64 -> 4.32) and the non-integer one above, the same boxes; --axis rewritten
        (t / "ax.json").write_text(json.dumps({"control_points": [{"z": 0.0, "y": 80.0, "x": 85.0}, {"z": 199.0, "y": 70.5, "x": 90.25}]}))
        for Uc in (U0 / 2, U):
            fc = Uc / U0; od = f"c{int(Uc * 100)}"
            out = run(BIN, "ct-resample", t / "s", "v", t / "o" / od, "--um", U0, "--to", Uc, "--level", 0, "--shard", 128, "--threads", 3, "--mode", "cubic",
                      "--axis", t / "ax.json", t / f"ax{od}.json", *sum((["--box", ",".join(map(str, b))] for b in boxes), []))
            print(out.strip())
            lvn = next(d.name for d in (t / "o" / od).iterdir() if d.is_dir())
            osc = [int(np.ceil(n / fc - 1e-9)) for n in sh]
            cb = ct_ = 0
            for b in boxes:
                lo = [int(np.floor(b[d] / fc + 1e-9)) for d in range(3)]
                hi = [min(osc[d], int(np.ceil((b[d] + b[3 + d]) / fc - 1e-9))) for d in range(3)]
                R = cref(V, (*lo, *hi), fc)
                C = read(t / "o", f"{od}/{lvn}", lo, [hi[d] - lo[d] for d in range(3)], td)
                dif = np.abs(C.astype(int) - R.astype(int))
                assert dif.max() <= 1, f"cubic {Uc} box {b}: max |diff| {dif.max()}"
                cb += int((dif > 0).sum()); ct_ += dif.size
            assert cb <= 1e-4 * ct_, f"cubic {Uc}: {cb} of {ct_} voxels differ by 1"
            ax = json.loads((t / f"ax{od}.json").read_text())["control_points"]
            k = U0 / Uc
            for q, r in zip(ax, [(0.0, 80.0, 85.0), (199.0, 70.5, 90.25)]):
                assert all(abs(q[c] - ((v + 0.5) * k - 0.5)) < 1e-3 for c, v in zip("zyx", r)), (q, r)
            print(f"PASS: ct-resample --mode cubic 8.64 -> {Uc:g} um, {ct_} voxels, {cb} rounding-tie diffs; --axis x {k:.4g}")
        print(f"PASS: ct-resample 8.64 -> 4.8 um, 2 boxes, {tot} voxels, {bad} rounding-tie diffs")


main()

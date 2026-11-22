"""ct-resample: a CT level trilinear to another voxel size over boxes (8.64 -> 4.8 um, f = 0.5556, centres aligned, edges clamped,
floor(v + 0.5)) against a numpy reference inside both boxes (|diff| <= 1 at rounding ties only, <= 0.01 % of voxels), absent outside the
boxes' shards (0), and the output group's OME scale 4.8."""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

if importlib.util.find_spec("numpy") is None:
    print("SKIP: no python numpy (the reference needs it)")
    sys.exit(0)
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
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
        print(f"PASS: ct-resample 8.64 -> 4.8 um, 2 boxes, {tot} voxels, {bad} rounding-tie diffs")


main()

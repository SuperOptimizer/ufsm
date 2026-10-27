"""ink-blend: two constant uint8 volumes on offset tile grids blended with sin^2 tile weights (lossless), plus the mean-pooled level."""
import json
import math
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
N, T = 128, 64   # 128^3 volumes, tiles of 64 on grids (0,0,0) and (32,32,32)


def run(*a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 0, r.stdout
    return r.stdout


with tempfile.TemporaryDirectory(prefix="ufsm-iblend-") as tmp:
    t = pathlib.Path(tmp)
    (t / "lut.txt").write_text(" ".join(str(min(v, 254)) for v in range(256)))
    ins = []
    for name, val in (("a", 100), ("b", 200)):
        lv = t / f"{name}.zarr" / "0"
        lv.mkdir(parents=True)
        (lv / ".zarray").write_text(json.dumps({"shape": [N] * 3, "chunks": [N] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                                "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
        (lv / "0" / "0").mkdir(parents=True)
        (lv / "0" / "0" / "0").write_bytes(bytes([val]) * N ** 3)
        run(BIN, "ingest-labels", t, f"{name}.zarr", t / name, "--um", "9.6", "--lut", t / "lut.txt", "--levels", "1", "--shard", "128", "--threads", "2")
        ins.append(t / name)
    run(BIN, "ink-blend", t / "out", "--um", "9.6", "--tile", f"{T},{T},{T}", "--grids", "0,0,0:32,32,32", *ins, "--q", "0", "--levels", "2", "--shard", "128", "--threads", "2")
    run(BIN, "read", t / "out", "9.6", 0, 0, 0, N, N, N, t / "o.raw")
    o = (t / "o.raw").read_bytes()
    at = lambda z, y, x: o[(z * N + y) * N + x]

    def expect(z, y, x):
        w = []
        for g in (0, 32):
            p = 1.0
            for c in (z, y, x):
                u = ((c + 0.5 - g) % T) / T
                p *= max(math.sin(math.pi * u) ** 2, 1e-6)
            w.append(p)
        return (100 * w[0] + 200 * w[1]) / (w[0] + w[1])

    for z, y, x in [(0, 0, 0), (32, 32, 32), (16, 16, 16), (31, 64, 5), (100, 7, 90)]:
        assert abs(at(z, y, x) - expect(z, y, x)) <= 1, (z, y, x, at(z, y, x), expect(z, y, x))
    assert at(0, 0, 0) >= 199 and at(32, 32, 32) <= 101
    run(BIN, "read", t / "out", "19.2", 0, 0, 0, 1, 1, 1, t / "c.raw")
    c = (t / "c.raw").read_bytes()[0]
    m = sum(at(z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)) / 8
    assert abs(c - m) <= 1, (c, m)
print("test_ink_blend: ok")

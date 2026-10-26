"""ingest-labels --src-level / --lut: a soft teacher map (zarr v2, level K) -> prob254 pyramid, absent chunks = lut[0], absent shards unknown."""
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "build/ufsm"
N, C = 64, 32          # level-1 array 64^3, chunks 32^3


def val(z, y, x):
    return (z * 7 + y * 3 + x) % 256


with tempfile.TemporaryDirectory(prefix="ufsm-lut-") as tmp:
    t = pathlib.Path(tmp)
    src = t / "pred.zarr"
    lv0 = src / "0"   # decoy level 0 (must not be read)
    lv0.mkdir(parents=True)
    (lv0 / ".zarray").write_text(json.dumps({"shape": [2 * N] * 3, "chunks": [C] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                             "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
    lv1 = src / "1"
    lv1.mkdir()
    (lv1 / ".zarray").write_text(json.dumps({"shape": [N] * 3, "chunks": [C] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                             "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
    present = [(0, 0, 0), (1, 1, 0)]   # chunk (0,0,1) etc. absent
    for cz, cy, cx in present:
        d = lv1 / str(cz) / str(cy)
        d.mkdir(parents=True, exist_ok=True)
        b = bytes(val(cz * C + z, cy * C + y, cx * C + x) for z in range(C) for y in range(C) for x in range(C))
        (d / str(cx)).write_bytes(b)
    lut = [min(254, v) if v >= 20 else 0 for v in range(256)]
    lut[0] = 0
    (t / "lut.txt").write_text(" ".join(map(str, lut)))
    out = t / "out"
    r = subprocess.run([BIN, "ingest-labels", t, "pred.zarr", out, "--um", "9.6", "--src-level", "1", "--lut", t / "lut.txt",
                        "--levels", "2", "--shard", "32", "--threads", "2"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert r.returncode == 0, r.stdout
    assert "64 x 64 x 64" in r.stdout, r.stdout   # level 1 was read, not the 128^3 decoy
    raw = t / "a.raw"
    r = subprocess.run([BIN, "read", out, "9.6", 0, 0, 0, N, N, N, raw], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert r.returncode == 0, r.stdout
    a = raw.read_bytes()
    assert len(a) == N ** 3
    at = lambda z, y, x: a[(z * N + y) * N + x]
    for z, y, x in [(0, 0, 0), (3, 5, 30), (31, 31, 31), (40, 33, 2), (63, 63, 31)]:
        assert at(z, y, x) == lut[val(z, y, x)], (z, y, x, at(z, y, x), lut[val(z, y, x)])
    assert at(0, 0, 40) == 255, at(0, 0, 40)   # whole shard absent -> unknown
    meta = json.loads((out / "9.6" / "zarr.json").read_text())
    assert "prob254" in json.dumps(meta), meta
    r = subprocess.run([BIN, "read", out, "19.2", 0, 0, 0, N // 2, N // 2, N // 2, t / "b.raw"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert r.returncode == 0, r.stdout
    b = (t / "b.raw").read_bytes()
    exp = max(lut[val(z, y, x)] for z in (0, 1) for y in (0, 1) for x in (0, 1))
    assert b[0] == exp, (b[0], exp)
print("test_ingest_lut: ok")

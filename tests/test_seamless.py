"""predict --seamless 2 (src/seamless.c) without a GPU: a stand-in pass (UFSM_SEAM_EXE) checks the arguments each pass gets and writes a
constant --heads store (pass a 100, pass b 200, lossless); the blend must give (100 wa + 200 wb) / (wa + wb) with wa, wb the sin^2 tile
weights of the grids 0 and S/2 (S = window - 2 halo = 256), the pass logs and .seam must be gone, and a second head (fib/H) is blended too."""
import json, math, os, pathlib, stat, subprocess, sys, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / os.environ.get("UFSM_TEST_BIN", "build/ufsm")
N, W, H = 256, 272, 8
S = W - 2 * H; S2 = S // 2

FAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
a = sys.argv; out = pathlib.Path(a[5]); o = {a[i]: a[i + 1] for i in range(6, len(a) - 1) if a[i].startswith("--")}
log = pathlib.Path(os.environ["SEAM_TEST_LOG"])
with log.open("a") as f: f.write(json.dumps({"out": str(out), "opts": o, "rest": a[1:6]}) + "\n")
assert os.environ.get("UFSM_SEAM_CHILD") == "1" and "--seamless" not in a and "--gpus" not in a
val = 100 if o["--grid-origin"] == "0,0,0" else 200
t = out.parent / ("raw_" + out.name); lv = t / "src.zarr" / "0"; lv.mkdir(parents=True, exist_ok=True)
lv.joinpath(".zarray").write_text(json.dumps({"shape": [%d] * 3, "chunks": [%d] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                             "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
for c in range(8): (lv / str(c >> 2) / str(c >> 1 & 1)).mkdir(parents=True, exist_ok=True); (lv / str(c >> 2) / str(c >> 1 & 1) / str(c & 1)).write_bytes(bytes([val]) * %d ** 3)
(t / "lut.txt").write_text(" ".join(str(min(v, 254)) for v in range(256)))
for head in ("ink", "fib/H"):
    subprocess.run([os.environ["SEAM_TEST_BIN"], "ingest-labels", str(t), "src.zarr", str(out / head), "--um", "9.6", "--lut", str(t / "lut.txt"),
                    "--levels", "1", "--shard", "128", "--threads", "2"], check=True, stdout=subprocess.DEVNULL)
''' % (N, N // 2, N // 2)


def run(env, *a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300, env=env)
    return r.returncode, r.stdout


with tempfile.TemporaryDirectory(prefix="ufsm-seam-", delete=not os.environ.get("SEAM_TEST_KEEP")) as tmp:
    t = pathlib.Path(tmp); fake = t / "fakepass.py"; fake.write_text(FAKE); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, UFSM_SEAM_EXE=str(fake), SEAM_TEST_LOG=str(t / "argv.log"), SEAM_TEST_BIN=str(BIN))
    out = t / "out"
    rc, txt = run(env, BIN, "predict", "ck.ckpt", "root", "key", out, "--um", "2.4", "--level", "2", "--heads", "ink,fib", "--seamless", "2",
                  "--window", W, "--halo", H, "--gpus", "0,1", "--q", "0", "--levels", "2", "--shard", "128", "--resume", "1", "--threads", "2")
    assert rc == 0, txt
    calls = [json.loads(l) for l in (t / "argv.log").read_text().splitlines()]
    assert len(calls) == 2, calls
    for c in calls:
        o = c["opts"]
        assert o["--window"] == str(W) and o["--halo"] == str(H) and o["--shard"] == str(S2) and o["--full-grid"] == "1", o
        assert o["--q"] == "0" and o["--levels"] == "1" and "--resume" not in o and o["--heads"] == "ink,fib" and o["--um"] == "2.4", o
    assert sorted(c["opts"]["--grid-origin"] for c in calls) == ["0,0,0", f"{S2},{S2},{S2}"], calls
    assert sorted(c["opts"]["--gpu"] for c in calls) == ["0", "1"], calls   # --gpus 0,1: one pass per GPU
    assert not (out / ".seam").exists(), "the passes must be removed"
    u = (np.arange(N) + 0.5)
    wa1 = np.maximum(np.sin(np.pi * (u % S) / S) ** 2, 1e-6); wb1 = np.maximum(np.sin(np.pi * ((u - S2) % S) / S) ** 2, 1e-6)
    for head in ("ink", "fib/H"):
        rr = t / f"{head.replace('/', '_')}.raw"
        rc, txt = run(env, BIN, "read", out / head, "9.6", 0, 0, 0, N, N, N, rr, "--threads", "2"); assert rc == 0, txt
        v = np.fromfile(rr, np.uint8).reshape(N, N, N).astype(np.float64)
        for z, y, x in ((0, 0, 0), (5, 77, 128), (128, 128, 128), (200, 31, 255), (64, 192, 100)):
            wa = wa1[z] * wa1[y] * wa1[x]; wb = wb1[z] * wb1[y] * wb1[x]; e = (100 * wa + 200 * wb) / (wa + wb)
            assert abs(v[z, y, x] - e) <= 1.0, (head, z, y, x, v[z, y, x], e)
        assert (out / head / "19.2").exists() or any(p.name.startswith("19.2") for p in (out / head).iterdir()), list((out / head).iterdir())
        meta = json.loads((out / head / "9.6" / "zarr.json").read_text())
        assert meta["attributes"]["ufsm"]["seamless"]["tile"] == S, meta["attributes"]
    # bad tile: window - 2 halo not a multiple of 256
    rc, txt = run(env, BIN, "predict", "ck.ckpt", "root", "key", t / "bad", "--heads", "ink", "--seamless", "2", "--window", 300, "--halo", 8)
    assert rc == 2 and "multiple of 256" in txt, txt
print("test_seamless: ok")

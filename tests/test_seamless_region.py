"""predict --seamless 2's blend region (perf 11-26; PHerc1203 r24: ink-seam off0 0.25, off256 9.32) without a GPU. A stand-in pass
(UFSM_SEAM_EXE) writes what predict writes: a smooth field plus a per-window offset (0 / 30 / 60 by the window's index on the pass's
grid: every window border a step), every shard of each window it computes (one window per 2^3 shards of S / 2), over its box rounded
to the shards, windows that meet a --tile-list tile only, and no all-zero shard. The blend must hide every window border: ink-seam
~0 at both offsets (0: pass a's borders, S / 2: pass b's), with no pass-b-alone voxels beyond the box / the tiles (they kept pass b's
borders, jump ~15-30), and a pass's absent (all-zero) shard counts as zeros with its sin^2 weight, not as weight 0."""
import json, os, pathlib, stat, subprocess, sys, tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / os.environ.get("UFSM_TEST_BIN", "build/ufsm")
N, W, H = 768, 272, 8
S = W - 2 * H; S2 = S // 2; C = 128   # tile 256, pass shards 128 (= the source chunks)

FAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
import numpy as np
N, S, C = %d, %d, %d
a = sys.argv; out = pathlib.Path(a[5]); o = {a[i]: a[i + 1] for i in range(6, len(a) - 1) if a[i].startswith("--")}
g = int(o["--grid-origin"].split(",")[0]); sh = int(o["--shard"]); assert sh == C, o
bx = [int(v) for v in o["--box"].split(",")]
lo = [max(0, bx[d] // sh * sh) for d in range(3)]; hi = [min(N, -(-(bx[d] + bx[3 + d]) // sh) * sh) for d in range(3)]
ns = N // sh; keep = np.zeros((ns,) * 3, bool)
if "--tile-list" in o:   # predict's pred_tile_keep: the shards meeting a listed tile
    tls = int(o["--tile-list-size"]); tlo = [int(v) for v in o["--tile-list-origin"].split(",")]
    for l in open(o["--tile-list"]):
        t = [int(v) for v in l.split()]
        s0 = [(tlo[d] + t[d] * tls) // sh for d in range(3)]; s1 = [-(-(tlo[d] + (t[d] + 1) * tls) // sh) for d in range(3)]
        keep[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]] = True
else: keep[:] = True
inbox = np.zeros_like(keep); inbox[lo[0] // sh:hi[0] // sh, lo[1] // sh:hi[1] // sh, lo[2] // sh:hi[2] // sh] = True; keep &= inbox
wr = np.zeros_like(keep)   # every shard of a window that has a kept shard (predict's super-shard windows), inside the box
for s in zip(*np.nonzero(keep)):
    w0 = [(s[d] * sh - g) // S * S + g for d in range(3)]
    wr[max(0, w0[0] // sh):(w0[0] + S) // sh, max(0, w0[1] // sh):(w0[1] + S) // sh, max(0, w0[2] // sh):(w0[2] + S) // sh] = True
wr &= inbox
zero = os.environ.get("SEAM_ZERO", "")   # "PASS:z,y,x": that pass writes the shard as zeros (absent)
t = out.parent / ("raw_" + out.name); lv = t / "src.zarr" / "0"; lv.mkdir(parents=True, exist_ok=True)
lv.joinpath(".zarray").write_text(json.dumps({"shape": [N] * 3, "chunks": [C] * 3, "dtype": "|u1", "fill_value": 0, "order": "C",
                                             "filters": None, "compressor": None, "zarr_format": 2, "dimension_separator": "/"}))
def field(z0, y0, x0):
    z, y, x = [np.arange(v, v + C, dtype=np.float64) for v in (z0, y0, x0)]   # (separable: broadcast, not a meshgrid)
    base = (70 + 15 * np.sin(z / 97.0))[:, None, None] + (15 * np.sin(y / 131.0 + 1))[None, :, None] + (15 * np.sin(x / 113.0 + 2))[None, None, :]
    k = ((np.floor((z - g) / S) * 7)[:, None, None] + (np.floor((y - g) / S) * 3)[None, :, None] + np.floor((x - g) / S)[None, None, :]) %% 3
    return np.clip(base + 30 * k, 1, 254).astype(np.uint8)
p = "a" if g == 0 else "b"
for s in zip(*np.nonzero(wr)):
    if zero == "%%s:%%d,%%d,%%d" %% (p, *s): continue
    d = lv / str(s[0]) / str(s[1]); d.mkdir(parents=True, exist_ok=True); (d / str(s[2])).write_bytes(field(*(v * C for v in s)).tobytes())
(t / "lut.txt").write_text(" ".join(str(min(v, 254)) for v in range(256)))
subprocess.run([os.environ["SEAM_TEST_BIN"], "ingest-labels", str(t), "src.zarr", str(out / "ink"), "--um", "9.6", "--lut", str(t / "lut.txt"),
                "--levels", "1", "--shard", str(C), "--threads", "4"], check=True, stdout=subprocess.DEVNULL)
m = out / "ink" / "9.6" / "zarr.json"; j = json.loads(m.read_text()); j["fill_value"] = 0; m.write_text(json.dumps(j))   # (a label store's fill is 255; predict's 0)
''' % (N, S, C)


def run(env, *a):
    r = subprocess.run([str(v) for v in a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600, env=env)
    return r.returncode, r.stdout


def seam(env, out, off):
    rc, txt = run(env, BIN, "ink-seam", out / "ink", "--um", "9.6", "--period", S, "--offset", off, "--n", 16, "--threads", 4)
    assert rc == 0, txt
    jumps = [max(abs(float(v)) for v in l.split("jump")[1].replace("z", "").replace("y", "").replace("x", "").split()) for l in txt.splitlines() if l.startswith("ink-seam: corner")]
    mean = float(txt.split("mean jump")[1].split()[0])
    return jumps, mean, txt


def predict(env, out, *extra):
    rc, txt = run(env, BIN, "predict", "ck.ckpt", "root", "key", out, "--um", "2.4", "--level", "2", "--heads", "ink", "--seamless", "2",
                  "--window", W, "--halo", H, "--q", "0", "--levels", "1", "--shard", S, "--threads", 4, *extra)
    assert rc == 0, txt
    return txt


with tempfile.TemporaryDirectory(prefix="ufsm-seamreg-", delete=not os.environ.get("SEAM_TEST_KEEP")) as tmp:
    t = pathlib.Path(tmp); fake = t / "fakepass.py"; fake.write_text(FAKE); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, UFSM_SEAM_EXE=str(fake), SEAM_TEST_BIN=str(BIN))
    # 1: a box (z, y, x 256..768): pass b's box starts S / 2 lower (128); its windows' shards there are outside the region
    # tiles {1,2}^3 + 0,1,1 (no box: pass b starts at S / 2, z 0..128 of tile 0,1,1 from pass a alone); pass b's windows spill into
    # every neighbouring tile
    TL = [(z, y, x) for z in (1, 2) for y in (1, 2) for x in (1, 2)] + [(0, 1, 1)]
    tiles = t / "tiles.txt"; tiles.write_text("".join(f"{z} {y} {x}\n" for z, y, x in TL))
    cases = {"box": ("--box", "256,256,256,512,512,512"), "tiles": ("--tile-list", tiles, "--tile-list-size", S)}
    for name, extra in cases.items():
        out = t / name; predict(env, out, *extra)
        for off in (0, S2):
            j, m, txt = seam(env, out, off)
            assert j and max(j) < 2.0 and abs(m) < 1.0, (name, off, txt)   # (before the fix: box off 128 corners at 128 jump ~25)
        rr = t / f"{name}.raw"; rc, txt = run(env, BIN, "read", out / "ink", "9.6", 0, 0, 0, N, N, N, rr, "--threads", 4); assert rc == 0, txt
        v = np.fromfile(rr, np.uint8).reshape(N, N, N)
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    b = v[z * S:(z + 1) * S, y * S:(y + 1) * S, x * S:(x + 1) * S]; inr = (z, y, x) in TL if name == "tiles" else min(z, y, x) >= 1
                    assert b.all() if inr else not b.any(), (name, z, y, x, "the region whole" if inr else "pass b alone beyond the region")
    # 2: pass a's shard 3,3,3 (voxels 384..512) all zero, so not written: its zeros blend with weight wa (before: weight 0, pass b alone)
    out = t / "zero"; predict(dict(env, SEAM_ZERO="a:3,3,3"), out, *cases["box"])
    ref = np.fromfile(t / "box.raw", np.uint8).reshape(N, N, N).astype(np.float64)
    rr = t / "zero.raw"; rc, txt = run(env, BIN, "read", out / "ink", "9.6", 0, 0, 0, N, N, N, rr, "--threads", 4); assert rc == 0, txt
    v = np.fromfile(rr, np.uint8).reshape(N, N, N).astype(np.float64)
    u = np.arange(N) + 0.5; wa1 = np.maximum(np.sin(np.pi * (u % S) / S) ** 2, 1e-6); wb1 = np.maximum(np.sin(np.pi * ((u - S2) % S) / S) ** 2, 1e-6)
    for z, y, x in ((400, 400, 400), (448, 448, 448), (390, 500, 460), (511, 384, 400)):
        wa = wa1[z] * wa1[y] * wa1[x]; wb = wb1[z] * wb1[y] * wb1[x]
        # the box run's value is (A wa + B wb) / (wa + wb); with A = 0 it is B wb / (wa + wb): B from the fake's field on grid S / 2
        zz, yy, xx = float(z), float(y), float(x)
        base = 70 + 15 * np.sin(zz / 97.0) + 15 * np.sin(yy / 131.0 + 1) + 15 * np.sin(xx / 113.0 + 2)
        k = (np.floor((zz - S2) / S) * 7 + np.floor((yy - S2) / S) * 3 + np.floor((xx - S2) / S)) % 3
        B = float(np.clip(base + 30 * k, 1, 254).astype(np.uint8)); e = B * wb / (wa + wb)
        assert abs(v[z, y, x] - e) <= 1.0, (z, y, x, v[z, y, x], e, ref[z, y, x])
    assert (v[:384] == ref[:384]).all() and (v[512:] == ref[512:]).all(), "outside the zero shard: the box run's bytes"
print("test_seamless_region: ok")

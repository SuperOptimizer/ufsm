#!/usr/bin/env python3
"""ufsm ink-ext --no-overlap: tiles at stride = tile size with blending off, so every output pixel is exactly its one tile's
prediction. Hecate 9.6 um on a synthetic 16-layer stack (--layers, 192 x 128, an exact 3 x 2 tiling) against --raw-in of the
same six tiles: |output - round(255 sigmoid(logit))| <= 1 everywhere. Needs numpy, the hecate_9.6um.pth weights and a GPU."""
import os, pathlib, struct, subprocess, sys, tempfile, zlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ink_ext_parity as P

W = P.DATA / "weights/hecate_9.6um.pth"
try:
    import numpy as np
except ImportError:
    P.skip("numpy not installed")
for p in (W, P.UFSM):
    if not pathlib.Path(p).exists():
        P.skip(f"{p} missing")
try:
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        P.skip("no GPU")
except OSError:
    P.skip("no GPU")


def write_tif(path, a):   # uint8 single strip, uncompressed
    h, w = a.shape
    tags = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8 + 2 + 12 * 9 + 4),
            (277, 3, 1, 1), (278, 3, 1, h), (279, 4, 1, w * h)]
    with open(path, "wb") as f:
        f.write(b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(tags)))
        for t, ty, n, v in tags:
            f.write(struct.pack("<HHI", t, ty, n) + (struct.pack("<HH", v, 0) if ty == 3 else struct.pack("<I", v)))
        f.write(struct.pack("<I", 0)); f.write(a.astype(np.uint8).tobytes())


def read_tif(path):   # tiffw's uint8 strips (deflate or none)
    b = open(path, "rb").read(); off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; tg = {}
    for i in range(n):
        t, ty, c = struct.unpack_from("<HHI", b, off + 2 + 12 * i); v = off + 2 + 12 * i + 8
        vals = [struct.unpack_from("<H" if ty == 3 else "<I", b, v)[0]] if c == 1 else list(struct.unpack_from(f"<{c}I", b, struct.unpack_from("<I", b, v)[0]))
        tg[t] = vals
    w, h = tg[256][0], tg[257][0]; data = b""
    for o, l in zip(tg[273], tg[279]):
        s = b[o:o + l]; data += zlib.decompress(s) if tg[259][0] == 8 else s
    return np.frombuffer(data, np.uint8)[: w * h].reshape(h, w)


rng = np.random.default_rng(3)
D, H, Wd, T = 16, 128, 192, 64
stack = np.clip(90 + 40 * rng.standard_normal((D, H, Wd)), 1, 255).astype(np.uint8)
with tempfile.TemporaryDirectory(prefix="ufsm-inkext-tiling-") as d:
    d = pathlib.Path(d); (d / "layers").mkdir()
    for i in range(D):
        write_tif(d / "layers" / f"{i:02d}.tif", stack[i])
    r = subprocess.run([P.UFSM, "ink-ext", "--model", "hecate96", "--weights", W, "--layers", d / "layers", "--no-overlap", "--out", d / "o", "--name", "t", "--batch", "4"],
                       capture_output=True, text=True, timeout=600)
    sys.stderr.write(r.stderr)
    assert r.returncode == 0, "ink-ext --layers --no-overlap failed"
    out = read_tif(d / "o" / "t_hecate96.tif")
    assert out.shape == (H, Wd), out.shape
    tiles = np.stack([stack[:, y:y + T, x:x + T] for y in range(0, H, T) for x in range(0, Wd, T)])
    logits = P.run_c("hecate96", W, tiles, batch=4)
    ref = np.zeros((H, Wd), np.float64); k = 0
    for y in range(0, H, T):
        for x in range(0, Wd, T):
            ref[y:y + T, x:x + T] = np.rint(255 / (1 + np.exp(-logits[k].astype(np.float64)))); k += 1
    err = np.abs(out.astype(np.int32) - ref.astype(np.int32))
    print(f"no-overlap tiling: max |out - tile prediction| {err.max()} (uint8), {len(tiles)} tiles")
    assert err.max() <= 1, "--no-overlap output is not the single-tile prediction"
print("test_ink_ext_tiling: OK")

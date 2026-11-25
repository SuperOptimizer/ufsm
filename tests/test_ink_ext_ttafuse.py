#!/usr/bin/env python3
"""ufsm ink-ext --tta-fuse (2026-11-25): with a non-mean fuse every --tta-views view runs as its own pass (one symmetry each) and each
pixel fuses the views. Identities against the in-pass mean (--tta-views N averages the views per tile before blending; linear, so equal up
to float rounding): median of 2 = trim of 2 = mean of 2, trim of 3 (floor(3 / 4) = 0 dropped) = mean of 3, each |d| <= 1 LSB. The median
and the logit mean of 3 differ from the mean of 3 (some pixel > 1 LSB) but stay close (< 64 LSB). --tta-fuse without --tta-views is refused.
Hecate 9.6 um on a synthetic 16-layer stack (--layers). Needs numpy, the hecate_9.6um.pth weights and a GPU."""
import pathlib, struct, subprocess, sys, tempfile, zlib
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


rng = np.random.default_rng(5)
D, H, Wd = 16, 160, 224
stack = np.clip(90 + 40 * rng.standard_normal((D, H, Wd)), 1, 255).astype(np.uint8)
stack[:, 40:90, 60:150] = np.clip(stack[:, 40:90, 60:150].astype(int) + 60, 0, 255).astype(np.uint8)   # a brighter block: structure for the views to disagree on
with tempfile.TemporaryDirectory(prefix="ufsm-inkext-ttafuse-") as d:
    d = pathlib.Path(d); (d / "layers").mkdir()
    for i in range(D):
        write_tif(d / "layers" / f"{i:02d}.tif", stack[i])

    def run(tag, *a):
        r = subprocess.run([str(P.UFSM), "ink-ext", "--model", "hecate96", "--weights", str(W), "--layers", str(d / "layers"), "--out", str(d / tag), "--name", "t",
                            "--batch", "4", *a], capture_output=True, text=True, timeout=600)
        sys.stderr.write(r.stderr[-1500:]); assert r.returncode == 0, f"ink-ext {a} failed"
        return read_tif(d / tag / "t_hecate96.tif").astype(np.int32)

    m2, m3 = run("m2", "--tta-views", "2"), run("m3", "--tta-views", "3")
    for tag, a, ref in (("med2", ("--tta-views", "2", "--tta-fuse", "median"), m2), ("trim2", ("--tta-views", "2", "--tta-fuse", "trim"), m2),
                        ("trim3", ("--tta-views", "3", "--tta-fuse", "trim"), m3)):
        e = np.abs(run(tag, *a) - ref).max(); print(f"{tag} vs the in-pass mean: max |d| {e}"); assert e <= 1, tag
    for tag in ("median", "logit"):
        e = np.abs(run(tag, "--tta-views", "3", "--tta-fuse", tag) - m3)
        print(f"{tag} of 3 vs mean of 3: max |d| {e.max()}, {np.mean(e > 1):.3f} of px > 1 LSB"); assert e.max() < 64 and (e > 1).any(), tag
    r = subprocess.run([str(P.UFSM), "ink-ext", "--model", "hecate96", "--weights", str(W), "--layers", str(d / "layers"), "--out", str(d / "bad"), "--tta-fuse", "median"],
                       capture_output=True, text=True)
    assert r.returncode == 2, "--tta-fuse without --tta-views must be refused"
print("test_ink_ext_ttafuse: OK")

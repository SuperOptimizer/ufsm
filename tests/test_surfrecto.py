#!/usr/bin/env python3
"""ufsm surfrecto-predict --raw-in (C/CUDA, fp16 storage) against the villa NetworkFromConfig build of the recto-surface U-Net
(tests/surfrecto_ref.py, fp32 CPU) on one real 2.4 um Paris 4 window: every encoder / decoder stage output (relative max |d|), the
logit margin, and the surface probability p = sigmoid(margin) (max / mean |d|, F1 agreement at 0.5). Then a small box through
the full predict path (window grid, z-score, Gaussian blend, sigmoid, shard writer) against a numpy re-implementation of villa's
blend from reference margins. Skips (exit 0) when torch, the checkpoint, the villa sources, the CT, the GPU or build/ufsm are missing.

CT: UFSM_SURFRECTO_CT root (default the volcomp mirror), UFSM_SURFRECTO_KEY (Paris 4 2.4 um level 0), UFSM_SURFRECTO_AT z,y,x,
UFSM_SURFRECTO_P window (default 128 in the suite; 256 = the model's patch, ~20 GB of CPU RAM for the reference).
Bounds (fp16 vs fp32, measured 2026-11-02 + margin) are printed with the numbers."""
import os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import surfrecto_ref as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
UFSM = ROOT / "build/ufsm"
CT_ROOT = os.environ.get("UFSM_SURFRECTO_CT", "https://dl.ash2txt.org/community-uploads/forrest/volcomp")
CT_KEY = os.environ.get("UFSM_SURFRECTO_KEY", "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0")
AT = [int(v) for v in os.environ.get("UFSM_SURFRECTO_AT", "39904,15000,15000").split(",")]
P = int(os.environ.get("UFSM_SURFRECTO_P", "128"))
CACHE = pathlib.Path(os.environ.get("UFSM_SURFRECTO_CACHE", str(R.DATA / "cache"))).expanduser()


def skip(msg):
    print(f"SKIP: {msg}"); sys.exit(0)


def main():
    try:
        import numpy as np, torch
    except ImportError:
        skip("torch / numpy missing")
    for p in [R.CKPT, R.SRC / "vesuvius/models/build/build_network_from_config.py", UFSM]:
        if not pathlib.Path(p).exists():
            skip(f"{p} missing")
    if "GPU" not in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout:
        skip("no GPU")
    torch.set_num_threads(int(os.environ.get("UFSM_TEST_TORCH_THREADS", "8")))
    CACHE.mkdir(parents=True, exist_ok=True)
    fails = 0
    with tempfile.TemporaryDirectory(dir=os.environ.get("UFSM_TEST_TMP", str(pathlib.Path.home() / "ufsm-data"))) as td:
        td = pathlib.Path(td)
        raw = td / "win.u8"
        r = subprocess.run([str(UFSM), "read", CT_ROOT, CT_KEY, *map(str, AT), str(P), str(P), str(P), str(raw), "--cache", str(CACHE)],
                           capture_output=True, text=True)
        if r.returncode:
            skip(f"CT read failed: {r.stderr[-300:]}")
        u = np.fromfile(raw, np.uint8).reshape(P, P, P)
        out = td / "out"
        r = subprocess.run([str(UFSM), "surfrecto-predict", "--ckpt", str(R.CKPT), "--raw-in", str(raw), "--raw-out", str(out),
                            "--window", str(P), "--keep", "1"], capture_output=True, text=True)
        print(r.stderr.strip())
        if r.returncode:
            print("FAIL: ufsm surfrecto-predict --raw-in failed"); return 1
        net, ck = R.build()
        ref, feats = R.run_window(net, u, keep=True)

        # stats
        m0, s0 = [float(v) for v in (out / "stats.txt").read_text().split()]
        x = u.astype(np.float32)
        dm, ds = abs(m0 - float(np.mean(x))), abs(s0 - float(np.std(x)))
        print(f"z-score stats: mean |d| {dm:.2e}, std |d| {ds:.2e}")
        if dm > 1e-3 or ds > 1e-3:
            print("FAIL: z-score statistics"); fails += 1
        # stages
        for k in range(13):
            nm = f"enc{k}" if k < 7 else f"dec{k - 7}"
            f = list(out.glob(f"{nm}_*.f32"))
            if not f or nm not in feats:
                print(f"FAIL: stage {nm} missing"); fails += 1; continue
            a = np.fromfile(f[0], np.float32).reshape(feats[nm].shape)
            b = feats[nm]
            rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            cos = float((a64 * b64).sum() / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
            print(f"{nm:5s} {str(b.shape):22s} rel max |d| {rel:.2e}  cos {cos:.6f}")
            if rel > 0.05 or cos < 0.999:
                print(f"FAIL: {nm}"); fails += 1
        mg = np.fromfile(out / "margin.f32", np.float32).reshape(P, P, P)
        dmg = np.abs(mg - ref)
        pa, pb = 1 / (1 + np.exp(-mg.astype(np.float64))), 1 / (1 + np.exp(-ref.astype(np.float64)))
        dp = np.abs(pa - pb)
        ba, bb = pa > 0.5, pb > 0.5
        f1 = 2 * (ba & bb).sum() / max(ba.sum() + bb.sum(), 1)
        u8d = np.abs(np.rint(255 * pa) - np.rint(255 * pb))
        print(f"margin |d| max {dmg.max():.4f} mean {dmg.mean():.5f} (|ref| max {np.abs(ref).max():.2f}); p |d| max {dp.max():.5f} mean {dp.mean():.2e};"
              f" uint8 |d| max {u8d.max():.0f}, {100 * (u8d > 1).mean():.4f}% > 1; F1@0.5 {f1:.5f} (surface fraction {bb.mean():.4f})")
        if dp.max() > 0.05 or dp.mean() > 2e-3 or f1 < 0.995:
            print("FAIL: probabilities"); fails += 1
        fails += box_check(td, np)
    print("PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


def grid(b0, b1, V, P, S):
    """surfrecto.c grid_of: stride S grid centred on [b0 - m, b1 + m), m = (P - S) / 2, clamped into [0, V - P], deduplicated"""
    m = (P - S) // 2; span = b1 - b0 + 2 * m
    n = 1 if span <= P else (span - P + S - 1) // S + 1
    total = (n - 1) * S + P; c0 = b0 - m - (total - span) // 2   # span, total >= 0: floor division = C division
    out = []
    for k in range(n):
        s = min(max(c0 + k * S, 0), V - P)
        if not out or out[-1] != s: out.append(s)
    return out


def box_check(td, np):
    """the predict path on a small box (window 128, overlap 0.5) vs numpy blending of the windows' margins it dumps
    (UFSM_SURFRECTO_DUMP), the dumped first window vs --raw-in on the same CT, zeros outside the box"""
    Pw, S = 128, 64
    z0, y0, x0 = AT[0] + 40, AT[1] + 40, AT[2] + 40
    box = (z0, y0, x0, z0 + 64, y0 + 192, x0 + 160)
    dump = td / "dump"; dump.mkdir()
    outz = td / "boxout"
    env = dict(os.environ, UFSM_SURFRECTO_DUMP=str(dump))
    r = subprocess.run([str(UFSM), "surfrecto-predict", str(R.CKPT), CT_ROOT, CT_KEY, str(outz), "--box", ",".join(map(str, box)),
                        "--window", str(Pw), "--overlap", "0.5", "--threads", "4", "--cache", str(CACHE)], capture_output=True, text=True, env=env)
    print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
    if r.returncode:
        print(f"FAIL: predict on a box: {r.stderr[-500:]}"); return 1
    V = (75784, 32693, 32693)   # Paris 4 2.4 um level 0 (the default CT); only the clamps use it
    G = [grid(box[a], box[3 + a], V[a], Pw, S) for a in range(3)]
    want = {f"win_{a}_{b}_{c}.f32" for a in G[0] for b in G[1] for c in G[2]}
    have = {p.name for p in dump.glob("win_*.f32")}
    fails = 0
    if want != have:
        print(f"FAIL: window origins: {len(have)} dumped, {len(want)} expected, {len(want ^ have)} differ"); return 1
    lo = [box[a] - 8 for a in range(3)]; n = [box[3 + a] - box[a] + 16 for a in range(3)]
    acc = np.zeros(n, np.float64); wsum = np.zeros(n, np.float64)
    g1 = np.exp(-0.5 * (np.arange(Pw) - Pw // 2) ** 2 / (Pw / 8) ** 2)
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    for nm in sorted(have):
        o = [int(v) for v in nm[4:-4].split("_")]
        m = np.fromfile(dump / nm, np.float32).reshape(Pw, Pw, Pw).astype(np.float64)
        sl_d, sl_w = [], []
        for a in range(3):
            a0, a1 = max(o[a], box[a]), min(o[a] + Pw, box[3 + a])
            sl_d.append(slice(a0 - lo[a], a1 - lo[a])); sl_w.append(slice(a0 - o[a], a1 - o[a]))
        acc[tuple(sl_d)] += (g3 * m)[tuple(sl_w)]; wsum[tuple(sl_d)] += g3[tuple(sl_w)]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 1 / (1 + np.exp(-np.clip(acc / wsum, -60, 60)))
    ref = np.where(wsum > 0, np.rint(255 * p), 0)
    raw = td / "box.u8"
    r = subprocess.run([str(UFSM), "read", str(outz), "2.4", *map(str, lo), *map(str, n), str(raw)], capture_output=True, text=True)
    if r.returncode:
        print(f"FAIL: reading the output: {r.stderr[-300:]}"); return 1
    got = np.fromfile(raw, np.uint8).reshape(n).astype(np.float64)
    d = np.abs(got - ref)
    print(f"box {box}: {len(have)} windows; uint8 vs numpy blend |d| max {d.max():.0f}, {100 * (d > 0).mean():.4f}% differ; "
          f"outside-box max {max(got[:8].max(), got[-8:].max(), got[:, :8].max(), got[:, -8:].max(), got[:, :, :8].max(), got[:, :, -8:].max()):.0f}, inside mean {got[8:-8, 8:-8, 8:-8].mean():.1f}")
    if d.max() > 1 or (d > 0).mean() > 1e-3:
        print("FAIL: blend"); fails += 1
    # --part 0/2 + 1/2 (128^3 shards: two shard rows in y) into one array == the whole-box run (same global grid, no seam)
    outp = td / "parts"
    for i in range(2):
        r = subprocess.run([str(UFSM), "surfrecto-predict", str(R.CKPT), CT_ROOT, CT_KEY, str(outp), "--box", ",".join(map(str, box)),
                            "--window", str(Pw), "--overlap", "0.5", "--threads", "4", "--cache", str(CACHE), "--shard", "128", "--part", f"{i}/2"],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"FAIL: predict --part {i}/2: {r.stderr[-400:]}"); return fails + 1
    rawp = td / "parts.u8"
    subprocess.run([str(UFSM), "read", str(outp), "2.4", *map(str, lo), *map(str, n), str(rawp)], capture_output=True)
    gp = np.fromfile(rawp, np.uint8).reshape(n).astype(np.float64)
    dpt = np.abs(gp - got)
    print(f"--part 0/2 + 1/2 vs the whole box: uint8 |d| max {dpt.max():.0f}, {100 * (dpt > 0).mean():.4f}% differ")
    if dpt.max() > 2 or (dpt > 0).mean() > 0.02:   # GPU run-to-run noise only (margin |d| ~0.05 flips ~0.6% of the roundings)
        print("FAIL: parts"); fails += 1
    # the first window: dumped margin vs --raw-in of the same CT
    o = [int(v) for v in sorted(have)[0][4:-4].split("_")]
    w = td / "w0.u8"
    subprocess.run([str(UFSM), "read", CT_ROOT, CT_KEY, *map(str, o), str(Pw), str(Pw), str(Pw), str(w), "--cache", str(CACHE)], capture_output=True)
    r = subprocess.run([str(UFSM), "surfrecto-predict", "--ckpt", str(R.CKPT), "--raw-in", str(w), "--raw-out", str(td / "w0"), "--window", str(Pw)],
                       capture_output=True, text=True)
    a = np.fromfile(td / "w0/margin.f32", np.float32); b = np.fromfile(dump / sorted(have)[0], np.float32)
    dd = float(np.abs(a - b).max()) if a.size == b.size else 1e9
    print(f"first window {o}: dumped margin vs --raw-in |d| max {dd:.2e}")
    if dd > 0.2:   # not bit-exact: xnet GroupNorm statistics use atomics (run-to-run margin |d| ~0.05, the fp16 noise level)
        print("FAIL: window extraction"); fails += 1
    return fails


if __name__ == "__main__":
    sys.exit(main())

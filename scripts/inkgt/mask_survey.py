#!/usr/bin/env python3
"""mask_survey.py OUT.tsv [--ufsm BIN] [--sources S.json] [--boxes 600] [--box 64] [--only NAME]: how much papyrus the CT masks clip.

Two measures, CPU only (ufsm read), on the -masked CT stores (masked voxels are exactly 0):
 1. label-free, every CT store under --ct-root: at the level nearest 38 um, the mask boundary (CT > 0 voxels with a CT == 0 6-neighbour,
    z end slices excluded) and the share of it that is papyrus-bright (>= the Otsu threshold of the CT > 0 voxels), against the same share
    over the whole inside. A mask in air has a dark boundary (ratio << 1); a mask cutting through papyrus has a boundary as bright as the
    inside (ratio ~1). Also the boundary voxels per inside voxel.
 2. labels, every training source with a label raster (--sources): --boxes random boxes of --box^3 label voxels, the labelled voxels
    (codes: 1..254; phase rasters: 0..251) and how many of them sit on CT == 0 (the CT at the level of the label's voxel size).
Rows: kind, name, ct, level, um, numbers. Untrusted inputs: none (our own stores).
"""
import argparse, json, math, os, re, subprocess, sys, tempfile
import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--ufsm", default="/home/forrest/ufsm/build/ufsm"); ap.add_argument("--ct-root", default="/vesuvius/ufsm/v2/ct")
ap.add_argument("--sources", default="/vesuvius/ufsm/univ/scratch1/sources-scratch1.json")
ap.add_argument("--boxes", type=int, default=600); ap.add_argument("--box", type=int, default=64); ap.add_argument("--only", default="")
ap.add_argument("--skip-ct", action="store_true"); ap.add_argument("--skip-labels", action="store_true")
a = ap.parse_args()
tmp = tempfile.mkdtemp(prefix="masksurvey-", dir="/vesuvius/ufsm/gt/pub1667-cmp")

def info(root, key):
    r = subprocess.run([a.ufsm, "info", root, key], capture_output=True, text=True).stdout
    lv = {}
    for m in re.finditer(r"^\s+(\d+)\s+([0-9.]+) um\s+shape (\d+) x (\d+) x (\d+)", r, re.M): lv[int(m.group(1))] = (float(m.group(2)), tuple(int(m.group(i)) for i in (3, 4, 5)))
    if not lv:
        m = re.search(r"shape (\d+) x (\d+) x (\d+)", r)
        if m: lv[-1] = (0.0, tuple(int(m.group(i)) for i in (1, 2, 3)))
    return lv

def read(root, key, z, y, x, nz, ny, nx):
    p = os.path.join(tmp, "r.raw")
    if subprocess.run([a.ufsm, "read", root, key, str(z), str(y), str(x), str(nz), str(ny), str(nx), p, "--threads", "8"], capture_output=True).returncode: return None
    v = np.fromfile(p, np.uint8); os.remove(p)
    return v.reshape(nz, ny, nx) if v.size == nz * ny * nx else None

def ero6(m):   # 6-neighbour binary erosion by shifts (scipy's was ~10x slower on GB volumes); outside = False
    e = m.copy(); e[0] = e[-1] = False; e[:, 0] = e[:, -1] = False; e[:, :, 0] = e[:, :, -1] = False
    e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]; e[:, :, 1:] &= m[:, :, :-1]; e[:, :, :-1] &= m[:, :, 1:]
    return e

def otsu(v):
    h = np.bincount(v, minlength=256).astype(np.float64); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    with np.errstate(all="ignore"): s = (mt * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))

rows = []
def emit(r): rows.append(r); print("\t".join(str(x) for x in r), flush=True)
SKIP = open(os.environ["MS_SKIP"]).read() if os.environ.get("MS_SKIP") else ""
if not a.skip_ct:
    for sc in sorted(os.listdir(a.ct_root)):
        if not sc.startswith("PHerc") or (a.only and a.only not in sc): continue
        for vol in sorted(os.listdir(os.path.join(a.ct_root, sc))):
            key = f"{sc}/{vol}"
            if f"\t{sc}\t{vol}\t" in SKIP: continue   # MS_SKIP: rows of an earlier (killed) run's log
            lv = info(a.ct_root, key)
            if not lv: continue
            L = min(lv, key=lambda k: abs(math.log(lv[k][0] / float(os.environ.get("MS_UM", "64")))) if lv[k][0] > 0 else 9); um, sh = lv[L]
            # z slabs of 256 (+1 halo slice each side): histograms of the inside, the boundary and the ring one voxel in
            st = ndimage.generate_binary_structure(3, 1); hin = np.zeros(256); hb = np.zeros(256); hb2 = np.zeros(256); ins = 0; nbd = 0; zlo = zhi = None; S = 128
            for z0 in range(0, sh[0], S):
                za, zb = max(0, z0 - 2), min(sh[0], z0 + S + 2); ct = read(a.ct_root, f"{key}/{L}", za, 0, 0, zb - za, sh[1], sh[2])
                if ct is None: break
                m = ct > 0; er = ero6(m); er2 = ero6(er); er3 = ero6(er2)
                c0, c1 = z0 - za, z0 - za + min(S, sh[0] - z0); mm, bb, bb2, cc = m[c0:c1], (m & ~er)[c0:c1], (er2 & ~er3)[c0:c1], ct[c0:c1]   # (bb2: the ring 2 voxels in, past the downsampling's edge blur)
                zz = np.flatnonzero(mm.any((1, 2)))
                if len(zz): zlo = z0 + zz[0] if zlo is None else zlo; zhi = z0 + zz[-1]
                hin += np.bincount(cc[mm], minlength=256); ins += int(mm.sum())
                # (z end slices of the object are mask boundary by the crop: dropped below by the slab-wise z extent check)
                keep = np.ones(c1 - c0, bool); zabs = np.arange(z0, z0 + c1 - c0)
                hb_s = [np.bincount(cc[k][bb[k]], minlength=256) for k in range(c1 - c0)]; hb2_s = [np.bincount(cc[k][bb2[k]], minlength=256) for k in range(c1 - c0)]
                if not hasattr(a, "_zh"): a._zh = []
                a._zh.append((zabs, hb_s, hb2_s)); nbd += int(bb.sum())
                del ct, m, er, er2, er3
            if not ins: emit(("ct", sc, vol, L, um, "empty")); a._zh = []; continue
            hb = np.zeros(256); hb2 = np.zeros(256)
            for zabs, hs, hs2 in a._zh:
                for k, zv in enumerate(zabs):
                    if zlo + 2 <= zv <= zhi - 2: hb += hs[k]; hb2 += hs2[k]
            a._zh = []
            p = hin / hin.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
            with np.errstate(all="ignore"): t = int(np.nanargmax((mu[-1] * w - mu) ** 2 / (w * (1 - w))))
            br_in = hin[t:].sum() / hin.sum(); br_b = hb[t:].sum() / max(hb.sum(), 1); br_b2 = hb2[t:].sum() / max(hb2.sum(), 1)
            emit(("ct", sc, vol, L, round(um, 2), f"inside {ins / (sh[0] * sh[1] * sh[2]):.3f}", f"otsu {t}", f"bright inside {br_in:.3f}",
                  f"bright boundary {br_b:.3f}", f"bright 2-in {br_b2:.3f}", f"clip ratio(2-in) {br_b2 / br_in:.2f}", f"boundary per inside {hb.sum() / ins:.4f}"))
if not a.skip_labels:
    J = json.load(open(a.sources)); rng = np.random.default_rng(1); seen = set()
    for s in J["sources"]:
        if a.only and a.only not in s["name"]: continue
        t = json.dumps(s.get("targets", {}))
        for root, key in sorted(set(re.findall(r'"root": "([^"]+)", "key": "([^"]+)"', t))):
            if (root, key, s["ct"]) in seen or "weight" in root: continue
            seen.add((root, key, s["ct"]))
            lab_um = float(key) if re.fullmatch(r"[0-9.]+", key) else None
            li = info(root, key);
            if not li or lab_um is None: emit(("label", s["name"], root, key, "no info")); continue
            sh = list(li.values())[0][1]; cl = info(s["root"], s["ct"])
            L = min(cl, key=lambda k: abs(math.log(cl[k][0] / lab_um)) if cl[k][0] > 0 else 9); cum, csh = cl[L]
            if abs(cum / lab_um - 1) > 0.05: emit(("label", s["name"], root, key, f"no CT level at {lab_um} um")); continue
            phase = "phase" in root.lower(); n = 0; nz = 0; nb = 0; B = a.box
            for _ in range(a.boxes):
                z, y, x = (int(rng.integers(0, max(1, sh[i] - B))) for i in range(3))
                lab = read(root, key, z, y, x, B, B, B)
                if lab is None: continue
                lm = (lab <= 251) if phase else ((lab > 0) & (lab < 255))
                if not lm.any(): continue
                ct = read(s["root"], f"{s['ct']}/{L}", z, y, x, B, B, B)
                if ct is None: continue
                nb += 1; n += int(lm.sum()); nz += int((lm & (ct == 0)).sum())
            emit(("label", s["name"], f"{root} {key}", L, round(cum, 2), f"boxes with labels {nb}", f"labelled voxels {n}", f"on CT 0 {nz / max(n, 1):.4f}"))
open(a.out, "w").write("\n".join("\t".join(str(x) for x in r) for r in rows) + "\n")
os.rmdir(tmp)

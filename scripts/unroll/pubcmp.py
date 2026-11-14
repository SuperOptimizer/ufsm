#!/usr/bin/env python3
"""pubcmp.py: label-free, CPU-only check of an unroll strip against a published unwrap (urender audit 2026-11-14 #1, #3).

  python3 -I scripts/unroll/pubcmp.py --flat O/strip/strip/flat --pub GT/pub1667-unwrap --banner banner-full.webp --out DIR
          [--renders RDIR] [--gate 1]

1. Correspondence: every banner pixel -> the published flattened grid (tifxyz_flattened, the coarse scan's voxels) -> the fine scan through
   the inverse of transforms/r2-transform.json's affine (held-out rms 0.77 vx on 1667) -> /4 (our level-2 grid) -> the nearest vertex of
   our stripes (flat/stripes/ws*/{x,y,z}.tif, cKDTree, <= 288 um). corr.npy: (stripe, row, col, d_um) per banner-grid pixel.
2. Sawtooth (direction gate, needs no image): along the published grid's columns (8-px steps, same stripe, d < 50 um) the local slope
   d(our column)/d(published column), against the global slope (regression of our position on the published position). A strip that runs
   the right way has both of one sign; the 11-14 1667 strip (dir +1, opposite handedness) had local +3.4 / global -3.5 = every turn
   traversed backwards. Also the share of steps that jump more than 3x the local pitch (sheet hops). sawtooth.jpg: the scatter.
3. With --renders (render's <s>.tif ink, <s>_ct.tif, fillmap/<s>_fill.tif): the published banner is registered to the grid (one of the
   4 flips by FFT cross-correlation of its papyrus mask with our CT, then a linear column drift from 1000-px windows; 1667: rot180, 467 px,
   x scale 0.981) and our ink / CT are sampled in its frame; AUC of our ink and of our CT against the published ink (banner < 128 vs
   192..239 papyrus) on pixels with our surface within 50 um and on the surface (fill flag 0). The published image is a model output at
   ~121 um/px, not GT; at chance level that does not matter.
Gate (--gate 1, exit 1 on failure): local and global slope of one sign, jumps <= 5 %; with renders also ink AUC >= CT AUC + 0.02.
Prints one 'pubcmp:' summary line; writes summary.json. 1667: ~7 min for the tree, ~12 GB.
"""
import argparse, glob, json, os, sys, time
import numpy as np, tifffile
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates, uniform_filter

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument('--flat', required=True); ap.add_argument('--pub', required=True); ap.add_argument('--banner', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--renders'); ap.add_argument('--gate', type=int, default=0)
ap.add_argument('--level-div', type=float, default=4.0, help='fine-scan voxels per our grid voxel (2.399 um scan, level 2: 4)')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--ink-gate', help='DIR1,DIR2: renders of p_recto / p_verso (u1667ink GATE=1: RDIR/c12, RDIR/c13); ink x max(p) / 255')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
log = lambda *s: print(f'pubcmp [{time.time() - t0:5.0f}s]', *s, flush=True)

# --- 1. correspondence
A = np.array(json.load(open(os.path.join(a.pub, 'transforms/r2-transform.json')))['transformation_matrix'])
Mi = np.linalg.inv(A[:, :3]); tr = A[:, 3]
fx, fy, fz = (tifffile.imread(os.path.join(a.pub, f'tifxyz_flattened/{c}.tif')) for c in 'xyz')
gh, gw = fx.shape; valid = (fx >= 0) & np.isfinite(fx)
ban = np.asarray(Image.open(a.banner).convert('L')); bh, bw = ban.shape
yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
gr = (yy + 0.5) * gh / bh - 0.5; gc = (xx + 0.5) * gw / bw - 0.5; del yy, xx
vm = map_coordinates(valid.astype(np.float32), [gr, gc], order=1) > 0.999
F = np.stack([map_coordinates(np.where(valid, v, 0), [gr[vm], gc[vm]], order=1) for v in (fx, fy, fz)], -1); del gr, gc
Q = ((F - tr) @ Mi.T) / a.level_div; del F
stripes = sorted(d for d in glob.glob(os.path.join(a.flat, 'stripes', '*')) if os.path.isdir(d))
st_tsv = os.path.join(a.flat, 'stripes.tsv'); col0 = {}
if os.path.exists(st_tsv):
    L = [l.rstrip('\n').split('\t') for l in open(st_tsv)]; h = L[0]
    for r in L[1:]: col0[r[0]] = int(r[h.index('out_col0')])
pts, idx, names = [], [], []
for si, d in enumerate(stripes):
    x, y, z = (tifffile.imread(os.path.join(d, f'{c}.tif')) for c in 'xyz'); v = (x > 0) & np.isfinite(x); r, c = np.nonzero(v)
    pts.append(np.stack([x[v], y[v], z[v]], -1).astype(np.float32)); idx.append(np.stack([np.full(r.size, si, np.int32), r, c], -1).astype(np.int32))
    names.append(os.path.basename(d)); del x, y, z
pts = np.concatenate(pts); idx = np.concatenate(idx)
log(f'{vm.sum()} banner-grid px on the published surface, {pts.shape[0]} of our vertices in {len(names)} stripes')
dist, k = cKDTree(pts, leafsize=32, balanced_tree=False, compact_nodes=False).query(Q, k=1, distance_upper_bound=30.0, workers=a.workers)
del pts, Q
ok = np.isfinite(dist); corr = np.full((bh, bw, 4), -1, np.float32); o = np.full((vm.sum(), 4), -1, np.float32)
o[ok, :3] = idx[k[ok]]; o[ok, 3] = dist[ok] * 9.596 * a.level_div / 4.0; corr[vm] = o; del o, idx, k
np.save(os.path.join(a.out, 'corr.npy'), corr)
d = corr[..., 3]; m50 = (d >= 0) & (d < 50)
S = {'banner': [bw, bh], 'grid': [gw, gh], 'stripes': names, 'pub_px': int(vm.sum()), 'within_288um': float(ok.mean()),
     'within_50um': float(m50.sum() / max(vm.sum(), 1)), 'd_med_um': float(np.median(d[d >= 0])) if (d >= 0).any() else None}
log(f"our surface within 50 um of {S['within_50um']:.3f} of the published px (d med {S['d_med_um']:.1f} um)")

# --- 2. sawtooth: local vs global direction (grid frame: banner column c <-> published grid column c * gw / bw)
s, rr, cc = corr[..., 0], corr[..., 1], corr[..., 2]
oc = np.full(s.shape, np.nan, np.float32)
for i, n in enumerate(names): sel = s == i; oc[sel] = cc[sel] + col0.get(n, 0)
K = 8; pa = m50[:, :-K] & m50[:, K:] & (s[:, :-K] == s[:, K:])
dloc = (oc[:, K:] - oc[:, :-K])[pa] / K                 # our columns per published px, locally
pitch = np.median(np.abs(dloc)) if dloc.size else np.nan
jumps = float((np.abs(dloc) > 3 * pitch).mean()) if dloc.size else np.nan
loc = float(np.median(dloc[np.abs(dloc) <= 3 * pitch])) if dloc.size else np.nan
yv, xv = np.nonzero(m50); glob_slope = float(np.polyfit(xv.astype(np.float64), oc[yv, xv].astype(np.float64), 1)[0]) if xv.size > 100 else np.nan
S.update({'local_slope': loc, 'global_slope': glob_slope, 'jump_share': jumps})
saw_ok = np.isfinite(loc) and np.isfinite(glob_slope) and np.sign(loc) == np.sign(glob_slope) and jumps <= 0.05
S['sawtooth_ok'] = bool(saw_ok)
log(f'direction: local slope {loc:+.2f}, global {glob_slope:+.2f} our cols / published px, jumps {jumps:.3f} -> {"OK" if saw_ok else "FAIL (the strip runs against the published sheet)"}')
# the scatter: our strip position vs the published position (cm), 300k points
if xv.size:
    rng = np.random.default_rng(0); j = rng.choice(xv.size, min(300000, xv.size), replace=False)
    um_ours = 4 * 9.596 * a.level_div / 4.0; X = oc[yv[j], xv[j]] * um_ours * 1e-4; Y = xv[j] * (gw / bw) * 20 * 7.91 * 1e-4
    W, H, Lm, B = 900, 520, 60, 40; im = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(im)
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    for u, v in zip(Lm + (X - x0) / (x1 - x0 + 1e-9) * (W - Lm - 10), H - B - (Y - y0) / (y1 - y0 + 1e-9) * (H - B - 25)): dr.point((u, v), fill=(0, 0, 0))
    dr.rectangle([Lm, 25, W - 10, H - B], outline='black')
    dr.text((5, 5), f'y: published grid position (cm), x: our strip position (cm); local {loc:+.2f} global {glob_slope:+.2f} jumps {jumps:.3f}', fill='black')
    for t in np.linspace(x0, x1, 6): dr.text((Lm + (t - x0) / (x1 - x0 + 1e-9) * (W - Lm - 10) - 10, H - B + 5), f'{t:.0f}', fill='black')
    for t in np.linspace(y0, y1, 6): dr.text((5, H - B - (t - y0) / (y1 - y0 + 1e-9) * (H - B - 25) - 5), f'{t:.0f}', fill='black')
    im.save(os.path.join(a.out, 'sawtooth.jpg'), quality=85)

# --- 3. ink vs the published ink
gate_auc = True
if a.renders:
    def auc(sc, pos, neg):
        p, n = sc[pos], sc[neg]
        if p.size < 50 or n.size < 50: return float('nan')
        lo, hi = min(p.min(), n.min()), max(p.max(), n.max()) + 1e-6
        hp = np.bincount(((p - lo) / (hi - lo) * 4095).astype(int), minlength=4096).astype(np.float64)
        hn = np.bincount(((n - lo) / (hi - lo) * 4095).astype(int), minlength=4096).astype(np.float64)
        return float((hp * (np.cumsum(hn) - hn + 0.5 * hn)).sum() / (p.size * n.size))
    sel = corr[..., 0] >= 0; si = corr[..., 0][sel].astype(int); ri = corr[..., 1][sel].astype(int); ci = corr[..., 2][sel].astype(int)
    maps = {}
    for nm, pat in (('ink', '{}.tif'), ('ct', '{}_ct.tif'), ('off', 'fillmap/{}_fill.tif')):
        val = np.zeros(sel.sum(), np.float32)
        for i, n in enumerate(names):
            p = os.path.join(a.renders, pat.format(n))
            if not os.path.exists(p): continue
            im = tifffile.imread(p)
            if nm == 'ink' and a.ink_gate:   # papyrus-gated ink (the head is never supervised in air)
                g2 = np.maximum(*[tifffile.imread(os.path.join(gd, f'{n}.tif')) for gd in a.ink_gate.split(',')]).astype(np.float32) / 255
                im = (im * g2).astype(np.uint8); del g2
            hh, ww = im.shape[0] // 4, im.shape[1] // 4
            g = (im[:hh * 4, :ww * 4] != 0) if nm == 'off' else im[:hh * 4, :ww * 4]
            g = uniform_filter(g.reshape(hh, 4, ww, 4).astype(np.float32).mean((1, 3)), 3)   # 38.4 um grid -> ~115 um (the banner's 121)
            q = si == i; val[q] = g[np.clip(ri[q], 0, hh - 1), np.clip(ci[q], 0, ww - 1)]; del im, g
        mp = np.zeros((bh, bw), np.float32); mp[sel] = val; maps[nm] = mp
    # the banner -> grid registration: flip by FFT (papyrus mask vs our CT, centred), then a linear column drift
    Am = np.where(m50, maps['ct'], 0); Am = np.where(m50, Am - Am[m50].mean(), 0)
    def xc(Aa, Bb, lim=None):
        hh, ww = Aa.shape; c = np.fft.irfft2(np.fft.rfft2(Aa, (2 * hh, 2 * ww)) * np.conj(np.fft.rfft2(Bb, (2 * hh, 2 * ww))), (2 * hh, 2 * ww))
        if lim: c = np.roll(c, (hh, ww), (0, 1))[hh - lim[0]:hh + lim[0] + 1, ww - lim[1]:ww + lim[1] + 1]; i = np.unravel_index(np.argmax(c), c.shape); return c.max(), i[0] - lim[0], i[1] - lim[1]
        i = np.unravel_index(np.argmax(c), c.shape); return c.max(), *[(i[q] if i[q] < c.shape[q] // 2 else i[q] - c.shape[q]) for q in (0, 1)]
    best = None
    for fl in ('id', 'h', 'v', 'hv'):
        Bf = ban[:, ::-1] if 'h' in fl else ban; Bf = Bf[::-1] if 'v' in fl else Bf
        Bm = (Bf < 240).astype(np.float32); Bm -= Bm.mean(); v, dy, dx = xc(Am, Bm)
        if best is None or v > best[0]: best = (v, fl, dy, dx)
    _, fl, dy, dx = best
    Bf = ban[:, ::-1] if 'h' in fl else ban; Bf = Bf[::-1] if 'v' in fl else Bf
    B0 = np.full((bh, bw), 255, np.float32)
    ys, xs = slice(max(dy, 0), min(bh, bh + dy)), slice(max(dx, 0), min(bw, bw + dx)); B0[ys, xs] = Bf[max(-dy, 0):min(bh, bh - dy), max(-dx, 0):min(bw, bw - dx)]
    cs, ds, ws = [], [], []
    for c0 in range(0, bw - 999, 500):
        sl = slice(c0, c0 + 1000); mm = m50[:, sl]
        if mm.sum() < 20000: continue
        Aa = np.where(mm, maps['ct'][:, sl] - maps['ct'][:, sl][mm].mean(), 0); Bm = (B0[:, sl] < 240).astype(np.float32); Bm -= Bm.mean()
        v, _, ddx = xc(Aa, Bm, (60, 150)); cs.append(c0 + 500); ds.append(ddx); ws.append(v / (np.linalg.norm(Aa) * np.linalg.norm(Bm) + 1e-9))
    if len(cs) >= 4:
        ws = np.array(ws); keep = ws >= np.median(ws); p = np.polyfit(np.array(cs)[keep], np.array(ds)[keep], 1)
        dxc = np.polyval(p, np.arange(bw)).astype(np.float32); yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        B0 = map_coordinates(B0, [yy, xx - dxc[None, :]], order=1, cval=255); del yy, xx
    else: p = [0, 0]
    S['banner_registration'] = {'flip': fl, 'dy': int(dy), 'dx': int(dx), 'drift_per_px': float(p[0]), 'drift_at_0': float(p[1])}
    base = m50 & (B0 < 240) & (maps['off'] < 0.25); ink = B0 < 128; clean = (B0 >= 192) & (B0 < 240)
    S['auc_ink'] = auc(maps['ink'], base & ink, base & clean); S['auc_ct'] = auc(maps['ct'], base & ink, base & clean)
    S['n_ink'] = int((base & ink).sum()); S['n_clean'] = int((base & clean).sum())
    S['p_pub_papyrus_given_ours'] = float((B0[m50] < 240).mean())
    gate_auc = S['auc_ink'] >= S['auc_ct'] + 0.02
    log(f"banner: {fl} {dy:+d},{dx:+d} drift {p[0]:+.4f}/px; P(published papyrus | ours) {S['p_pub_papyrus_given_ours']:.3f}; "
        f"AUC vs the published ink: ink {S['auc_ink']:.3f}, CT {S['auc_ct']:.3f} ({S['n_ink']} ink px)")
    # a registered side-by-side of the densest published text (banner orientation)
    col = int(np.argmax(uniform_filter((base & ink).sum(0).astype(np.float32), 1400))); c0 = max(0, min(bw - 1400, col - 700))
    rows = [B0]
    for nm in ('ink', 'ct'):
        x = maps[nm]; v = x[m50]; lo, hi = np.percentile(v, [1, 99.5]) if v.size else (0, 255); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        rows.append(np.where(m50, 255 - x if nm == 'ink' else x, 128))
    unflip = (lambda z: z[::-1] if 'v' in fl else z); unh = (lambda z: z[:, ::-1] if 'h' in fl else z)
    img = np.concatenate([np.concatenate([unh(unflip(r))[:, :][:, :], np.zeros((6, bw))], 0) for r in rows], 0)
    cc0 = bw - 1400 - c0 if 'h' in fl else c0
    Image.fromarray(img[:, cc0:cc0 + 1400].astype(np.uint8)).save(os.path.join(a.out, 'side_by_side.jpg'), quality=82)
S['gate'] = bool(saw_ok and gate_auc)
json.dump(S, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
print('pubcmp: ' + ' '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in S.items() if k not in ('stripes', 'banner', 'grid')), flush=True)
sys.exit(1 if a.gate and not S['gate'] else 0)

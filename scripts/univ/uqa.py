#!/usr/bin/env python3
"""scripts/univ/uqa.py OUTDIR NAME UM PHASE.raw CT.raw N [--rv RV.raw] [--ink INK.raw] [--step 2]: label-free quality of one box (2026-11-12).

The universal net's unseen-scroll block (scripts/univ/ueval.sh): no labels anywhere on these scrolls, so every number is a self-consistency
check between the net's own outputs and the CT. Inputs are raw uint8 cubes of the box: PHASE = ufsm fit-phase-box of the box's fit
(floor(252 frac(W)), 255 = outside; one value per STEP^3 voxels, dims ceil(N / STEP)), CT / RV / INK at the box's native voxels (N^3 or
nz,ny,nx). RV = the rv head (1 + 254 (1 + p_recto - p_verso) / 2), INK = the ink head (prob254). Sheets are the level sets W = integer
(ufsm wraps), so the sheet band is frac(W) near the box's sheet phase phi (measured from the CT) and the gap near phi + 0.5.
Metrics (one tab row, header first, -> OUTDIR/uqa.tsv; a PNG of the mid-z slice -> OUTDIR/NAME.png):
  h1         first-harmonic amplitude of the CT along frac(W) / CT sd (W's level sets follow the CT sheets)      higher
  phi        the phase of the CT maximum (the sheets' frac W; the band / gap / rv sides below are relative to it)
  on_pap     share of sheet-band voxels (|frac W - phi| < 0.08) on papyrus (CT >= the box's Otsu threshold)            higher = sheets on papyrus
  gap_void   share of mid-gap voxels (|frac W - phi| > 0.35) off papyrus                                        higher = gaps in the voids
  contrast   (CT mean on the band - CT mean mid-gap) / CT sd                                                     higher
  sp_um      median sheet spacing 1 / |grad W| (um);  sp_cv  its robust spread (IQR / median) on papyrus      lower = regular spacing
  jumps      share of grid steps where W jumps by > 0.25 turn (the fit's tears / slips)                          lower
  nrm_deg    median angle between the sheet normals (grad W) of neighbouring grid points;  nrm30  share > 30 deg   lower = smooth
  rv_agree   (RV) share of confident rv voxels (|rv - 128| > 32) in the band whose side agrees with the phase: recto on the inner face
             (frac W just below phi: W grows outward), 0.5 = no relation, < 0.5 = the convention inverted          higher
  ink_air    (INK) share of ink (> 127) voxels off papyrus;  ink_band  share of ink voxels within the sheet band +- 0.15     lower / higher
"""
import os, sys
import numpy as np

def otsu(v):
    h = np.bincount(v.ravel(), minlength=256).astype(np.float64); p = h / h.sum(); w = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
    s = (mt * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12); return int(np.argmax(s))

def cube(path, n, step=1):
    d = [(x + step - 1) // step for x in n]; a = np.fromfile(path, np.uint8)
    if a.size != d[0] * d[1] * d[2]: sys.exit(f"uqa: {path}: {a.size} bytes, expected {d}")
    return a.reshape(d)

def pool(a, s, shape):   # mean over s^3 blocks (CT / heads to the phase grid; edge blocks padded by replication)
    if s == 1: return a.astype(np.float32)
    pz, py, px = (shape[0] * s - a.shape[0], shape[1] * s - a.shape[1], shape[2] * s - a.shape[2])
    a = np.pad(a, ((0, pz), (0, py), (0, px)), mode='edge').astype(np.float32)
    return a.reshape(shape[0], s, shape[1], s, shape[2], s).mean((1, 3, 5))

def wrap_png(d, name):   # uqa.py --wrap-png DIR NAME: ink | CT renders of the box's largest wrap side by side -> DIR/NAME_wrap.png
    import glob, tifffile
    from PIL import Image
    im = [tifffile.imread(g[0]) for g in (sorted(glob.glob(os.path.join(d, f'rw/{k}/w*.tif'))) for k in ('ink', 'ct')) if g and '_fill' not in g[0]]
    if not im: print('uqa: no wrap renders', file=sys.stderr); return
    h = min(i.shape[0] for i in im); Image.fromarray(np.concatenate([i[:h].astype(np.uint8) for i in im], 1)).save(os.path.join(d, name + '_wrap.png'))

def main():
    if sys.argv[1] == '--wrap-png': return wrap_png(sys.argv[2], sys.argv[3])
    a = sys.argv[1:]; kv = {}
    while len(a) > 6 and a[-2].startswith('--'): kv[a[-2]] = a[-1]; a = a[:-2]
    out, name, um, ph_p, ct_p, n = a[0], a[1], float(a[2]), a[3], a[4], [int(x) for x in a[5].split(',')]
    if len(n) == 1: n = n * 3
    S = int(kv.get('--step', 2)); os.makedirs(out, exist_ok=True)
    ph = cube(ph_p, n, S); ok = ph != 255; f = ph.astype(np.float32) / 252.0; g = ph.shape
    ctn = cube(ct_p, n); thr = otsu(ctn[ctn > 0]) if (ctn > 0).any() else 128; ct = pool(ctn, S, g); pap = ct >= thr
    # the sheets' phase: CT modulation by frac(W), first harmonic (the wraps W = integer sit on a sheet face, ~0.1 turn off its centre on
    # the wbench fits; measured per box, not assumed): h1 = its amplitude / CT sd (convention-free: how well W's level sets follow the
    # CT sheets), phi = the phase of the CT maximum. The band / gap / rv sides are taken relative to phi.
    cz = (ct[ok] - ct[ok].mean()) * np.exp(2j * np.pi * f[ok]); zc = cz.mean(); sd = float(ct[ok].std()) or 1.0
    phi = float(np.angle(zc) / (2 * np.pi)) % 1.0; e = (f - phi + 0.5) % 1.0 - 0.5; d0 = np.abs(e)
    band = ok & (d0 < 0.08); gap = ok & (d0 > 0.35)
    r = {'box': name, 'ok': round(float(ok.mean()), 4), 'thr': thr}
    r['on_pap'] = float(pap[band].mean()) if band.any() else np.nan
    r['gap_void'] = float((~pap[gap]).mean()) if gap.any() else np.nan
    r['h1'] = float(2 * abs(zc) / sd); r['phi'] = phi; r['contrast'] = float((ct[band].mean() - ct[gap].mean()) / sd) if band.any() and gap.any() else np.nan
    # grad W by wrapped differences (W jumps by one turn at the theta cut and frac wraps at sheets: differences mod 1 into [-0.5, 0.5))
    def dif(ax):
        x = np.diff(f, axis=ax); x = (x + 0.5) % 1.0 - 0.5; hi = [slice(None)] * 3; lo = [slice(None)] * 3; hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
        return x, ok[tuple(hi)] & ok[tuple(lo)]
    dz, mz = dif(0); dy, my = dif(1); dx, mx = dif(2)
    jumps = np.concatenate([np.abs(dz[mz]), np.abs(dy[my]), np.abs(dx[mx])]); r['jumps'] = float((jumps > 0.25).mean()) if jumps.size else np.nan
    G = np.stack([dz[:, :-1, :-1], dy[:-1, :, :-1], dx[:-1, :-1, :]]); M = mz[:, :-1, :-1] & my[:-1, :, :-1] & mx[:-1, :-1, :]
    M &= (np.abs(G) < 0.25).all(0)   # (no tears inside the cell)
    gn = np.sqrt((G ** 2).sum(0)); M &= gn > 1e-6
    sp = (S * um) / np.maximum(gn, 1e-9); pc = pap[:-1, :-1, :-1] & M
    if pc.any():
        q1, q2, q3 = np.percentile(sp[pc], [25, 50, 75]); r['sp_um'] = float(q2); r['sp_cv'] = float((q3 - q1) / q2)
    else: r['sp_um'] = r['sp_cv'] = np.nan
    N = G / np.maximum(gn, 1e-9); ang = []
    for ax in range(3):   # neighbouring normals along each axis
        hi = [slice(None)] * 3; lo = [slice(None)] * 3; hi[ax] = slice(1, None); lo[ax] = slice(0, -1); hi, lo = tuple(hi), tuple(lo)
        mm = M[hi] & M[lo]; c = np.clip(N[0][hi][mm] * N[0][lo][mm] + N[1][hi][mm] * N[1][lo][mm] + N[2][hi][mm] * N[2][lo][mm], -1, 1)
        ang.append(np.degrees(np.arccos(c)).astype(np.float32))
    ang = np.concatenate(ang); r['nrm_deg'] = float(np.median(ang)) if ang.size else np.nan; r['nrm30'] = float((ang > 30).mean()) if ang.size else np.nan
    r['rv_agree'] = r['ink_air'] = r['ink_band'] = np.nan
    if '--rv' in kv:
        rv = pool(cube(kv['--rv'], n), S, g); conf = band & (np.abs(rv - 128) > 32) & (rv > 0)
        inner = e < 0   # W below the sheet's phase = its inner side (W grows outward)
        if conf.any(): r['rv_agree'] = float(((rv > 128) == inner)[conf].mean())
    ink = None
    if '--ink' in kv:
        ink = pool(cube(kv['--ink'], n), S, g); ik = ok & (ink > 127) & (ink < 255)
        if ik.any(): r['ink_air'] = float((~pap[ik]).mean()); r['ink_band'] = float((d0[ik] < 0.15).mean())
    keys = ['box', 'ok', 'thr', 'h1', 'phi', 'on_pap', 'gap_void', 'contrast', 'sp_um', 'sp_cv', 'jumps', 'nrm_deg', 'nrm30', 'rv_agree', 'ink_air', 'ink_band']
    fn = os.path.join(out, 'uqa.tsv'); new = not os.path.exists(fn)
    with open(fn, 'a') as fo:
        if new: fo.write('\t'.join(keys) + '\n')
        fo.write('\t'.join(str(r[k]) if isinstance(r[k], (str, int)) else ('na' if np.isnan(r[k]) else f'{r[k]:.4f}') for k in keys) + '\n')
    print('\t'.join(f'{k} {r[k]}' if isinstance(r[k], (str, int)) else f'{k} {r[k]:.4f}' for k in keys))
    # the eyeball image: the mid-z CT slice (left), the same with the sheet band (green) and ink (red, when given) over it (right)
    try:
        from PIL import Image
        z = n[0] // 2; c = ctn[z].astype(np.uint8); zz = min(z // S, g[0] - 1)
        up = lambda m: np.kron(m, np.ones((S, S), bool))[:c.shape[0], :c.shape[1]]
        rgb = np.stack([c, c, c], -1).astype(np.float32)
        b = up(band[zz]); rgb[b] = rgb[b] * 0.5 + np.array([0, 255, 0]) * 0.5
        if ink is not None:
            ik = np.kron(ink[zz], np.ones((S, S)))[:c.shape[0], :c.shape[1]]; w = np.clip((ik - 64) / 191, 0, 1)[..., None]; rgb = rgb * (1 - w) + np.array([255, 0, 0]) * w
        Image.fromarray(np.concatenate([np.stack([c, c, c], -1), rgb.astype(np.uint8)], 1)).save(os.path.join(out, name + '.png'))
    except Exception as e: print(f'uqa: no image ({e})', file=sys.stderr)

main()

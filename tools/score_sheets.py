#!/usr/bin/env python3
"""Sheet-quality scoring of checkpoints on held-out Paris4 boxes (thickness, localisation and instances; 2026-10-09).

Per box and checkpoint (output channel 0 = recto unless --channel):
  pixel      best F1 / ROC AUC against the thin reference at native resolution over CT > 0 voxels (labels upsampled by the
             trainer's mapping; the same numbers as tools/score_surface.py)
  tolerance  on the 4.8 um label grid (tools/label_grid.py): best F1 at tolerance 0 / 1 / 2 label voxels (Euclidean ball)
             of the raw prediction (max over each label voxel's natives) and of its radial ridge (non-maximum suppression
             along the direction away from the scroll axis: a native voxel stays if it is >= the voxels 1 and 2 steps either
             way, the rule of nn_ridge_u8 without its widening)
  profile    the prediction sampled +-16 native voxels along the radial line through reference voxels (label centres):
             peak offset (outward > 0), full width at half maximum (native voxels), double-peak rate, mean profile
  bands      instance scores against the winding band reference (as tools/score_bands.py: VOI split/merge, ARAND, wraps
             split, pieces merged; pieces = connected voxels below the recto cutoff), over known voxels and over known
             material voxels (4.8 um CT mean >= tau; tau = Otsu over the box's CT > 0 voxels unless --material-tau)
  legacy     the label-grid scores again with the older {2j, 2j + 1} pooling of the tools before 2026-10-09
  side       (--side-channel C, train --side 1) accuracy of the recto/verso side and sheet pieces cut on its verso -> recto faces
  phase      (--phase-channel C, train --side 2: channels C = 0.5 + 0.5 cos, C + 1 = 0.5 + 0.5 sin of the winding phase)
             circular phase error (turns) against the reference phase (ufsm band --phase) over known voxels, the side it
             implies (phase < 0.5: recto side) and its accuracy, mean confidence (length of the (cos, sin) vector), and sheet
             pieces cut where the phase wraps (6-neighbours more than half a turn apart), over all CT and over material,
             and pieces cut at the phase wraps and the recto ridge together (best recto cutoff and confidence)
usage: score_sheets.py --out DIR NAME=CKPT [...] [--boxes z,y,x,nz,ny,nx;...] [--gpu 0] [--root R --ct G --cache C
       --labels L --codes Q --axis A] [--window 544,16] [--channel 0]
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_grid import label_box, pool_max, pool_mean, upsample
from score_bands import pieces, compare
from band_field import instances as ref_instances
from sheet_diagnostics import binary_scores

UFSM = Path(__file__).resolve().parent.parent / 'build/ufsm'
ROOT = '/vesuvius/usrm/volcomp'
CT = 'PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr'
LABELS = '/vesuvius/ufsm/gt/paris4-sheet-20261003/training-4p8.zarr'
CODES = '/vesuvius/ufsm/gt/paris4-band-20261005/q-4p8.zarr'
AXIS = '/vesuvius/ufsm/runs/noisy-labels-20261005/axis-0.json'
BOXES = '48128,17408,16384,384,1024,1024;48512,17408,16384,512,1024,1024'
TOLS = (0, 1, 2)
HALF = 16          # profile half length, native voxels
PEAK_WIN = 8       # the peak is searched within +-PEAK_WIN of the label centre


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(' '.join(map(str, cmd))[:300] + '\n' + r.stderr[-2000:])


def read(store, key, lo, size, path, cache=None):
    cmd = [UFSM, 'read', store, key, *lo, *size, path, '--threads', '16'] + (['--cache', cache] if cache else [])
    run(cmd)
    v = np.fromfile(path, np.uint8).reshape(tuple(int(s) for s in size)); Path(path).unlink(); return v


def predict(g, ck, box, out, channel):
    if (out / 'zarr.json').exists(): return
    run([UFSM, 'predict', ck, g.root, g.ct, out, '--um', '2.4', '--box', ','.join(map(str, box)), '--window', g.window[0], '--halo', g.window[1],
         '--shard', '512', '--grid-origin', '0,0,0', '--ema', '1', '--q', '8', '--threads', '16', '--axis', g.axis, '--levels', '1',
         '--gpu', g.gpu, '--channel', channel] + (['--cache', g.cache] if g.cache else []) + g.predict_args.split())


def axis_of(path):
    pts = np.array(sorted([[p['z'], p['y'], p['x']] for p in json.loads(Path(path).read_text())['control_points']]), float)
    return lambda z: (np.interp(z, pts[:, 0], pts[:, 1]), np.interp(z, pts[:, 0], pts[:, 2]))


def radial_nms(p, o, axis):
    """p (native box at o): keep voxels >= their radial neighbours at 1 and 2 steps (nearest-voxel sampling), else 0"""
    out = np.zeros_like(p); n = p.shape
    yy, xx = np.meshgrid(np.arange(n[1], dtype=np.float32), np.arange(n[2], dtype=np.float32), indexing='ij')
    for z in range(n[0]):
        cy, cx = axis(o[0] + z)
        dy = yy + (o[1] - cy); dx = xx + (o[2] - cx); r = np.maximum(np.hypot(dy, dx), 1e-3)
        uy, ux = dy / r, dx / r
        s = p[z]; keep = np.ones(s.shape, bool)
        for step in (1, 2):
            for sg in (1, -1):
                y = np.rint(yy + sg * step * uy).astype(np.int64); x = np.rint(xx + sg * step * ux).astype(np.int64)
                ok = (y >= 0) & (y < n[1]) & (x >= 0) & (x < n[2])
                nb = np.where(ok, s[np.clip(y, 0, n[1] - 1), np.clip(x, 0, n[2] - 1)], 0)
                keep &= s >= nb
        out[z] = np.where(keep, s, 0)
    return out


def ball(r):
    if r == 0: return np.ones((1, 1, 1), bool)
    a = np.arange(-r, r + 1); z, y, x = np.meshgrid(a, a, a, indexing='ij')
    return z * z + y * y + x * x <= r * r


def best_f1(tp_hist, all_hist, rec_hist, npos):
    """F1 at every cutoff 1..255 from histograms (precision counts per prediction byte, recall per dilated byte)"""
    tp = np.cumsum(tp_hist[::-1])[::-1].astype(float); al = np.cumsum(all_hist[::-1])[::-1].astype(float)
    rc = np.cumsum(rec_hist[::-1])[::-1].astype(float)
    p = tp / np.maximum(al, 1); r = rc / max(npos, 1); f = 2 * p * r / np.maximum(p + r, 1e-12)
    f[0] = 0; b = int(np.argmax(f))
    return dict(f1=float(f[b]), cutoff=b / 255, precision=float(p[b]), recall=float(r[b]))


def tolerance_scores(pl, gt, valid):
    """best F1 per tolerance of a label-grid prediction (uint8) against the thin reference"""
    out = {}; npos = int((gt & valid).sum())
    for t in TOLS:
        fp = ball(t)
        near_gt = ndi.binary_dilation(gt, fp) if t else gt
        pm = ndi.grey_dilation(pl, footprint=fp) if t else pl
        tp_hist = np.bincount(pl[valid & near_gt], minlength=256); all_hist = np.bincount(pl[valid], minlength=256)
        rec_hist = np.bincount(pm[valid & gt], minlength=256)
        out['tol%d' % t] = best_f1(tp_hist, all_hist, rec_hist, npos)
    return out


def profiles(p, o, gt, valid, axis, nmax=200000, seed=0):
    """radial profiles of the native prediction through reference label centres"""
    j = np.argwhere(gt & valid)
    if not len(j): return None
    rng = np.random.default_rng(seed)
    if len(j) > nmax: j = j[rng.choice(len(j), nmax, replace=False)]
    c = 2 * j   # native centre of label voxel (o // 2 + j) relative to o (o even)
    c = c[(c < np.array(p.shape)).all(1)]
    cy, cx = axis(o[0] + c[:, 0])
    dy = c[:, 1] + o[1] - cy; dx = c[:, 2] + o[2] - cx; r = np.maximum(np.hypot(dy, dx), 1e-3)
    uy, ux = dy / r, dx / r
    t = np.arange(-HALF, HALF + 1, dtype=np.float64)
    coords = np.stack([np.repeat(c[:, 0:1].astype(float), len(t), 1), c[:, 1:2] + t * uy[:, None], c[:, 2:3] + t * ux[:, None]])
    inb = ((coords[1] >= 0) & (coords[1] <= p.shape[1] - 1) & (coords[2] >= 0) & (coords[2] <= p.shape[2] - 1)).all(1)   # whole line in the box
    coords, c = coords[:, inb], c[inb]
    if not len(c): return None
    v = ndi.map_coordinates(p, coords.reshape(3, -1), order=1, mode='nearest', prefilter=False, output=np.float32).reshape(len(c), len(t))
    w = slice(HALF - PEAK_WIN, HALF + PEAK_WIN + 1)
    pk = np.argmax(v[:, w], 1) + HALF - PEAK_WIN; pv = v[np.arange(len(v)), pk]
    hit = pv >= 0.2 * 255
    fwhm = np.full(len(v), np.nan); dbl = np.zeros(len(v), bool)
    for i in np.nonzero(hit)[0]:
        row, k, h = v[i], pk[i], pv[i] / 2
        lft = k
        while lft > 0 and row[lft] >= h: lft -= 1
        rgt = k
        while rgt < len(row) - 1 and row[rgt] >= h: rgt += 1
        if row[lft] < h and row[rgt] < h:   # interpolated half-maximum crossings
            a = lft + (h - row[lft]) / max(row[lft + 1] - row[lft], 1e-6); b = rgt - (h - row[rgt]) / max(row[rgt - 1] - row[rgt], 1e-6)
            fwhm[i] = b - a
        loc = np.nonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:]))[0] + 1
        for m in loc:
            if abs(m - k) >= 3 and row[m] >= 0.5 * pv[i]:
                lo_, hi_ = sorted((m, k))
                if row[lo_:hi_ + 1].min() < 0.8 * row[m]: dbl[i] = True; break
    off = (pk - HALF)[hit]; fw = fwhm[hit]; fwv = fw[np.isfinite(fw)]
    return dict(samples=int(len(v)), detected=float(hit.mean()), offset_median=float(np.median(off)) if len(off) else None,
                offset_mean=float(off.mean()) if len(off) else None, offset_std=float(off.std()) if len(off) else None,
                abs_offset_mean=float(np.abs(off).mean()) if len(off) else None,
                fwhm_median=float(np.median(fwv)) if len(fwv) else None, fwhm_p25=float(np.percentile(fwv, 25)) if len(fwv) else None,
                fwhm_p75=float(np.percentile(fwv, 75)) if len(fwv) else None, fwhm_unbounded=float(1 - len(fwv) / max(len(fw), 1)),
                double_peak=float(dbl[hit].mean()) if hit.any() else None, mean_profile=[round(float(x), 1) for x in v.mean(0)])


def otsu(x):
    h = np.bincount(x, minlength=256).astype(float); w = np.cumsum(h); m = np.cumsum(h * np.arange(256))
    wb, mb = w[:-1], m[:-1]; wf = w[-1] - wb; mf = m[-1] - mb
    s = wb * wf * (mb / np.maximum(wb, 1) - mf / np.maximum(wf, 1)) ** 2
    return int(np.argmax(s)) + 1


def band_scores(pl, ref, known, material, inside, cutoffs):
    """instance scores per recto cutoff over known voxels and over known material voxels (the same pieces)"""
    kn, km = [], []
    for cut in cutoffs:
        lab = pieces(pl >= int(round(cut * 255)), inside)
        kn.append(dict(method='known recto>=%.3f' % cut, **compare(ref, lab, known)))
        km.append(dict(method='material recto>=%.3f' % cut, **compare(ref, lab, known & material)))
    return kn, km


MIN_PIECE = 2000   # label voxels (as tools/score_bands.py)


def side_cut(side_r, material, lo, axis, sigma=1.0, valid=None):
    """label-grid cut voxels: recto-side voxels with a verso-side material 6-neighbour on a verso -> recto boundary, i.e. one where
    the (smoothed) recto indicator increases outward along the radial direction: the recto faces between sheets. The other
    boundaries (recto -> verso going outward: midway between sheets) stay connected. valid: voxels whose side is known (the
    smoothing is normalised over them; default all)."""
    n = side_r.shape
    f = ndi.gaussian_filter(side_r.astype(np.float32), sigma)
    if valid is not None:   # normalised: the known voxels' recto share, extended smoothly into unknown neighbours
        f = ndi.gaussian_filter((side_r & valid).astype(np.float32), sigma) / np.maximum(ndi.gaussian_filter(valid.astype(np.float32), sigma), 1e-3)
    vm = material & ~side_r
    nbv = np.zeros(n, bool)   # has a verso-side material 6-neighbour
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(0, -1); b[ax] = slice(1, None)
        nbv[tuple(a)] |= vm[tuple(b)]; nbv[tuple(b)] |= vm[tuple(a)]
    cand = side_r & material & nbv
    cut = np.zeros(n, bool)
    yy, xx = np.meshgrid(np.arange(n[1], dtype=np.float32), np.arange(n[2], dtype=np.float32), indexing='ij')
    gy = np.gradient(f, axis=1); gx = np.gradient(f, axis=2)
    for z in range(n[0]):
        if not cand[z].any(): continue
        cy, cx = axis(2.0 * (lo[0] + z))
        dy = 2.0 * (yy + lo[1]) - cy; dx = 2.0 * (xx + lo[2]) - cx; r = np.maximum(np.hypot(dy, dx), 1e-3)
        cut[z] = cand[z] & (gy[z] * dy / r + gx[z] * dx / r > 0)
    return cut


def side_pieces(side_r, material, inside, lo, axis, conn=1, sigma=1.0, valid=None):
    """sheet instances from a recto/verso side map: connected material minus the cuts; small pieces dropped; every other
    voxel inside takes the nearest piece"""
    cut = side_cut(side_r, material, lo, axis, sigma, valid)
    lab, nl = ndi.label(material & ~cut, ndi.generate_binary_structure(3, conn))
    size = np.bincount(lab.ravel(), minlength=nl + 1); keep = size >= MIN_PIECE; keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    if (lab > 0).any():
        _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(inside, lab[tuple(ind)], 0)
    return lab, float(cut.sum() / max(material.sum(), 1))


def phase_of(c, s):
    """label-grid phase in turns [0, 1) and confidence [0, 1] from the cos / sin channels (bytes, 0.5 + 0.5 cos x 255)"""
    x = c.astype(np.float32) - 127.5; y = s.astype(np.float32) - 127.5
    return np.mod(np.arctan2(y, x) / (2 * np.pi), 1.0), np.minimum(np.hypot(x, y) / 127.5, 1.0)


def phase_cut(ph, domain):
    """wrap voxels: the low-phase voxel of every 6-neighbour pair inside the domain whose phases differ by more than half a
    turn (the recto face, where the phase falls from ~1 back to 0 going outward)"""
    cut = np.zeros(ph.shape, bool)
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(0, -1); b[ax] = slice(1, None); a = tuple(a); b = tuple(b)
        d = ph[b] - ph[a]; both = domain[a] & domain[b]
        cut[a] |= both & (d > 0.5); cut[b] |= both & (d < -0.5)
    return cut


def phase_pieces(ph, domain, inside, conn=1, extra=None):
    """sheet instances from the phase: connected domain voxels minus the wraps (and minus extra, e.g. the recto ridge); small
    pieces dropped; every other voxel inside takes the nearest piece"""
    cut = phase_cut(ph, domain)
    if extra is not None: cut |= extra
    lab, nl = ndi.label(domain & ~cut, ndi.generate_binary_structure(3, conn))
    size = np.bincount(lab.ravel(), minlength=nl + 1); keep = size >= MIN_PIECE; keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    if (lab > 0).any():
        _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(inside, lab[tuple(ind)], 0)
    return lab, float(cut.sum() / max(domain.sum(), 1))


PHASE_CONF = (0.0, 0.2, 0.4, 0.6)   # pieces are cut within voxels at least this confident (best VOI reported, like the recto cutoffs)


COMBO_CUT = (0.1, 0.2, 0.3, 0.5)   # recto cutoffs of the combined cut (recto ridge or phase wrap)


def phase_scores(ph, conf, pref, inside, material, ref, known, confs=PHASE_CONF, pl=None):
    pk = (pref < 252) & inside
    g = pref.astype(np.float32) / 252.0
    e = np.abs(ph - g); e = np.minimum(e, 1 - e)
    r = dict(known_fraction=float(pk.sum() / max(inside.sum(), 1)),
             error_mean=float(e[pk].mean()), error_median=float(np.median(e[pk])), error_mean_material=float(e[pk & material].mean()),
             within_0p1=float((e[pk] < 0.1).mean()),
             side_accuracy=float(((ph[pk] < 0.5) == (g[pk] < 0.5)).mean()), side_accuracy_material=float(((ph[pk & material] < 0.5) == (g[pk & material] < 0.5)).mean()),
             confidence_known=float(conf[pk].mean()), confidence_inside=float(conf[inside].mean()))
    for dom, nm in ((inside, 'all'), (material, 'material')):
        rr = []
        for c in confs:
            lab, cf = phase_pieces(ph, dom & (conf >= c), inside)
            rr.append(dict(confidence=c, cut_fraction=cf, known=compare(ref, lab, known), material=compare(ref, lab, known & material)))
        r['pieces_' + nm] = min(rr, key=lambda x: x['known']['voi'])
        r['pieces_' + nm + '_by_confidence'] = rr
    if pl is not None:   # combined: cut at the recto ridge (label-grid max >= cutoff) and at the phase wraps, within confident voxels
        rr = []
        for t in COMBO_CUT:
            rm = pl >= int(round(t * 255))
            for c in confs[1:]:
                lab, cf = phase_pieces(ph, inside & (conf >= c), inside, extra=rm)
                rr.append(dict(recto=t, confidence=c, cut_fraction=cf, known=compare(ref, lab, known), material=compare(ref, lab, known & material)))
        r['pieces_combo'] = min(rr, key=lambda x: x['known']['voi'])
        r['pieces_combo_all'] = rr
    return r


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--boxes', default=BOXES); a.add_argument('--gpu', default='0')
    a.add_argument('--root', default=ROOT); a.add_argument('--ct', default=CT); a.add_argument('--cache'); a.add_argument('--labels', default=LABELS)
    a.add_argument('--codes', default=CODES); a.add_argument('--axis', default=AXIS); a.add_argument('--window', default='544,16')
    a.add_argument('--channel', default='0'); a.add_argument('--material-tau', type=int)
    a.add_argument('--predict-args', default='', help='extra ufsm predict options (e.g. "--gn-frozen F")')
    a.add_argument('--quick', action='store_true', help='pixel and label-grid tolerance scores only')
    a.add_argument('--side-channel', type=int, help='output channel of the recto/verso side (train --side): sheet pieces from it')
    a.add_argument('--phase-channel', type=int, help='first of the two winding-phase channels (train --side 2): phase error and pieces')
    a.add_argument('--cutoffs', default='0.1,0.15,0.2,0.25,0.3,0.4,0.5'); a.add_argument('ckpts', nargs='+')
    g = a.parse_args()
    g.window = g.window.split(',')
    axis = axis_of(g.axis)
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    cutoffs = [float(c) for c in g.cutoffs.split(',')]
    rows = []
    for bi, bs in enumerate(g.boxes.split(';')):
        box = [int(v) for v in bs.split(',')]; o, n = np.array(box[:3]), np.array(box[3:])
        if (o % 2).any() or (n % 2).any(): raise SystemExit('boxes must have even origin and size')
        lo, ln = label_box(o, n)
        bd = out / ('box%d' % bi); bd.mkdir(exist_ok=True)
        ref_path = bd / 'ref.npz'
        if not ref_path.exists():
            ct = read(g.root, g.ct + '/0', o, n, bd / 'ct.raw', g.cache)
            inside = pool_max(ct, o) > 0; ctm = pool_mean(ct, o)
            tau = g.material_tau if g.material_tau is not None else otsu(ct[ct > 0])
            gt = read(g.labels, '4.8', lo, ln, bd / 'lab.raw') > 0
            run([UFSM, 'band', g.codes, '4.8', bs, bd / 'band.raw', '--axis', g.axis])
            k = np.fromfile(bd / 'band.raw', np.uint8).reshape(tuple(ln)).astype(np.int16); k[k == 255] = -1; (bd / 'band.raw').unlink()
            k[~inside] = -1
            lab, _ = ref_instances(k)
            ctv = ct > 0; del ct
            np.savez_compressed(ref_path, k=k, lab=lab, inside=inside, ctm=ctm.astype(np.float16), gt=gt, ctv=np.packbits(ctv), tau=tau)
        side_ref = bd / 'side_ref.npy'
        if g.side_channel is not None and not side_ref.exists():
            run([UFSM, 'band', g.codes, '4.8', bs, bd / 'band.raw', '--axis', g.axis, '--side', bd / 'side.raw'])
            np.save(side_ref, np.fromfile(bd / 'side.raw', np.uint8).reshape(tuple(ln))); (bd / 'side.raw').unlink(); (bd / 'band.raw').unlink()
        phase_ref = bd / 'phase_ref.npy'
        if g.phase_channel is not None and not phase_ref.exists():
            run([UFSM, 'band', g.codes, '4.8', bs, bd / 'band.raw', '--axis', g.axis, '--phase', bd / 'phase.raw'])
            np.save(phase_ref, np.fromfile(bd / 'phase.raw', np.uint8).reshape(tuple(ln))); (bd / 'phase.raw').unlink(); (bd / 'band.raw').unlink()
        R = np.load(ref_path)
        k, ref, inside, gt, tau = R['k'], R['lab'], R['inside'], R['gt'], int(R['tau'])
        material = inside & (R['ctm'].astype(np.float32) >= tau)
        ctv = np.unpackbits(R['ctv'])[:int(np.prod(n))].reshape(tuple(n)).astype(bool)
        known = (k >= 0) & (ref > 0)
        gt_n = upsample(gt, o, n)
        for spec in g.ckpts:
            name, ck = spec.split('=', 1)
            cd = bd / name; cd.mkdir(exist_ok=True)
            res = cd / 'sheets.json'
            if res.exists(): rows.append(json.load(open(res))); continue
            predict(g, ck, box, cd / ('ch' + g.channel), g.channel)
            p = read(cd / ('ch' + g.channel), '2.4', (0, 0, 0), n, cd / 'p.raw')
            pos = np.bincount(p[gt_n & ctv], minlength=256); neg = np.bincount(p[~gt_n & ctv], minlength=256)
            r = dict(name=name, ckpt=ck, box=box, tau=tau, pixel=binary_scores(pos, neg))
            pl = pool_max(p, o)
            r['tolerance_raw'] = tolerance_scores(pl, gt, inside)
            if g.quick: json.dump(r, open(res, 'w'), indent=1); rows.append(r); continue
            thin = radial_nms(p, o, axis)
            r['tolerance_ridge'] = tolerance_scores(pool_max(thin, o), gt, inside)
            r['tolerance_raw_legacy'] = tolerance_scores(pool_max(p, o, legacy=True), gt, inside)
            r['profile'] = profiles(p, o, gt, inside, axis)
            bk, bm = band_scores(pl, ref, known, material, inside, cutoffs)
            r['bands'] = {t: min(rr, key=lambda x: x['voi']) for t, rr in (('known', bk), ('material', bm))}
            cut = float(r['bands']['known']['method'].split('>=')[1])   # legacy pooling at the same cutoff
            r['bands']['legacy'] = dict(method='legacy recto>=%.3f' % cut, **compare(ref, pieces(pool_max(p, o, legacy=True) >= int(round(cut * 255)), inside), known))
            r['bands_all'] = bk + bm
            r['material_fraction_of_known'] = float((known & material).sum() / max(known.sum(), 1))
            if g.side_channel is not None:   # sheet pieces from the predicted side (label grid, mean over each label voxel's natives)
                sc = str(g.side_channel); predict(g, ck, box, cd / ('ch' + sc), sc)
                ps = pool_mean(read(cd / ('ch' + sc), '2.4', (0, 0, 0), n, cd / 'p.raw'), o) >= 127.5
                sref = np.load(side_ref); sk = (sref <= 1) & inside
                lab, cf = side_pieces(ps, material, inside, lo, axis)
                r['side'] = dict(accuracy_known=float((ps[sk] == (sref[sk] == 1)).mean()), accuracy_material=float((ps[sk & material] == (sref[sk & material] == 1)).mean()),
                                 cut_fraction=cf, known=compare(ref, lab, known), material=compare(ref, lab, known & material))
            if g.phase_channel is not None:   # winding phase (label grid, mean over each label voxel's natives)
                cs = []
                for c in (g.phase_channel, g.phase_channel + 1):
                    predict(g, ck, box, cd / ('ch%d' % c), str(c))
                    cs.append(pool_mean(read(cd / ('ch%d' % c), '2.4', (0, 0, 0), n, cd / 'p.raw'), o))
                ph, conf = phase_of(*cs)
                np.savez_compressed(cd / 'phase.npz', ph=(ph * 252).astype(np.uint8), conf=(conf * 255).astype(np.uint8))
                r['phase'] = phase_scores(ph, conf, np.load(phase_ref), inside, material, ref, known, pl=pl)
            json.dump(r, open(res, 'w'), indent=1); rows.append(r)
            del p, thin
    for r in rows:
        if 'bands' not in r:
            print('%-14s box%d F1 %.3f AUC %.3f cutoff %.2f mean p(surface) %.3f (background %.3f) | label-grid F1 %s' % (r['name'], r['box'][0],
                  r['pixel']['best_binary_f1'], r['pixel']['roc_auc'] or 0, r['pixel']['best_cutoff'], r['pixel']['mean_probability_surface'],
                  r['pixel']['mean_probability_background'], '/'.join('%.3f' % r['tolerance_raw']['tol%d' % i]['f1'] for i in TOLS)), flush=True)
            continue
        t, tr, pf, b = r['tolerance_raw'], r['tolerance_ridge'], r['profile'] or {}, r['bands']
        print('%-14s box%d F1 %.3f AUC %.3f | label-grid F1 raw %s ridge %s | FWHM %s offset %s+-%s dbl %s | VOI %.3f (m %.3f) material VOI %.3f (m %.3f)'
              % (r['name'], r['box'][0], r['pixel']['best_binary_f1'], r['pixel']['roc_auc'] or 0,
                 '/'.join('%.3f' % t['tol%d' % i]['f1'] for i in TOLS), '/'.join('%.3f' % tr['tol%d' % i]['f1'] for i in TOLS),
                 pf.get('fwhm_median'), pf.get('offset_mean') and round(pf['offset_mean'], 2), pf.get('offset_std') and round(pf['offset_std'], 2),
                 pf.get('double_peak') and round(pf['double_peak'], 3),
                 b['known']['voi'], b['known']['voi_merge'], b['material']['voi'], b['material']['voi_merge']), flush=True)
        if 'side' in r:
            sd = r['side']
            print('%-14s box%d side: accuracy %.3f (material %.3f) | pieces VOI %.3f (split %.3f merge %.3f) ARAND %.3f, material VOI %.3f (merge %.3f), split %d merged %d'
                  % (r['name'], r['box'][0], sd['accuracy_known'], sd['accuracy_material'], sd['known']['voi'], sd['known']['voi_split'], sd['known']['voi_merge'],
                     sd['known']['adjusted_rand'], sd['material']['voi'], sd['material']['voi_merge'], sd['known']['bands_split'], sd['known']['pieces_merged']), flush=True)
        if 'phase' in r:
            q = r['phase']; pa, pm = q['pieces_all'], q['pieces_material']
            print('%-14s box%d phase: error %.3f turns (median %.3f, material %.3f, <0.1 %.3f) side accuracy %.3f (material %.3f) confidence %.2f | '
                  'pieces all (conf >= %.1f) VOI %.3f (merge %.3f) material VOI %.3f (merge %.3f) | material pieces (conf >= %.1f) VOI %.3f (merge %.3f)'
                  % (r['name'], r['box'][0], q['error_mean'], q['error_median'], q['error_mean_material'], q['within_0p1'], q['side_accuracy'],
                     q['side_accuracy_material'], q['confidence_known'], pa['confidence'], pa['known']['voi'], pa['known']['voi_merge'], pa['material']['voi'],
                     pa['material']['voi_merge'], pm['confidence'], pm['known']['voi'], pm['known']['voi_merge']), flush=True)
            if 'pieces_combo' in q:
                pc = q['pieces_combo']
                print('%-14s box%d combined cut (recto >= %.1f or phase wrap, confidence >= %.1f): VOI %.3f (split %.3f merge %.3f) ARAND %.3f, material VOI %.3f'
                      % (r['name'], r['box'][0], pc['recto'], pc['confidence'], pc['known']['voi'], pc['known']['voi_split'], pc['known']['voi_merge'],
                         pc['known']['adjusted_rand'], pc['material']['voi']), flush=True)
    json.dump(rows, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

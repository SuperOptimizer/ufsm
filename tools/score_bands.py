#!/usr/bin/env python3
"""Instance (whole-sheet) scoring of checkpoints on held-out boxes against the band reference (task band_affinity).

Everything is compared on the 4.8 um label grid (the reference band field's grid; prediction max-pooled 2x where noted).
Reference instances: `ufsm band` on the box (winding_mod14 raster), connected across neighbours in the same band.
Predicted instances, two ways:
  aff   (band_affinity checkpoints) boundary = any offset-1 affinity (channels 1..3; or offset 8, channels 4..6) says
        "different" (< 0.5) in the 2x block; pieces = connected non-boundary voxels inside the CT
  recto (any checkpoint) boundary = recto probability (channel 0, max-pooled) >= cutoff; pieces as above
Pieces smaller than MIN_PIECE label voxels are dropped; dropped and boundary voxels take the nearest piece.
Metrics over voxels known in the reference: variation of information (split = H(pred|ref), merge = H(ref|pred), bits),
adjusted Rand, wraps split (reference band covered >= 5% by 2+ pieces), pieces merged (piece covering >= 5% of 2+ bands).
usage: score_bands.py --out DIR NAME=CKPT [...] [--boxes z,y,x,nz,ny,nx;...] [--gpu 0] [--cutoffs 0.3,0.4]
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_connectivity import read, UFSM
from band_field import instances as ref_instances

ROOT = '/vesuvius/usrm/volcomp'
CT = 'PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr'
CODES = '/vesuvius/ufsm/gt/paris4-band-20261005/q-4p8.zarr'
AXIS = '/vesuvius/ufsm/runs/noisy-labels-20261005/axis-0.json'
BOXES = '48128,17408,16384,384,1024,1024;48512,17408,16384,512,1024,1024'
MIN_PIECE = 2000
FULL = np.ones((3, 3, 3), bool)


def predict(ck, box, out, gpu, channel):
    if (out / 'zarr.json').exists(): return
    cmd = [str(UFSM), 'predict', ck, ROOT, CT, str(out), '--um', '2.4', '--box', ','.join(map(str, box)), '--window', '528', '--halo', '8',
           '--shard', '512', '--grid-origin', '0,0,0', '--ema', '1', '--q', '8', '--threads', '16', '--axis', AXIS, '--levels', '1',
           '--gpu', str(gpu), '--channel', str(channel)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-2000:])


def cout_of(ckpt):
    """output channels from the checkpoint header line (UFSM{...\"cout\":N,...})"""
    with open(ckpt, 'rb') as f: line = f.readline().decode('utf-8', 'replace')
    return int(line.split('"cout":')[1].split(',')[0])


def pool2max(v):
    s = tuple(n // 2 * 2 for n in v.shape); v = v[:s[0], :s[1], :s[2]]
    return v.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).max(axis=(1, 3, 5))


def pieces(boundary, inside):
    lab, n = ndi.label(inside & ~boundary, FULL)
    size = np.bincount(lab.ravel(), minlength=n + 1); keep = size >= MIN_PIECE; keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    if (lab > 0).any():   # every other voxel inside takes the nearest kept piece
        _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(inside, lab[tuple(ind)], 0)
    return lab


def compare(ref, pred, known):
    r = ref[known].astype(np.int64); p = pred[known].astype(np.int64)
    pairs, cnt = np.unique(r * (p.max() + 1) + p, return_counts=True)
    ri, pi = pairs // (p.max() + 1), pairs % (p.max() + 1)
    n = cnt.sum(); pr = np.bincount(ri, weights=cnt); pp = np.bincount(pi, weights=cnt)
    pij = cnt / n
    h_ref_given_pred = -np.sum(pij * np.log2(cnt / pp[pi]))   # merge part
    h_pred_given_ref = -np.sum(pij * np.log2(cnt / pr[ri]))   # split part
    comb = lambda x: x * (x - 1) / 2.0
    sij, sa, sb, tot = comb(cnt).sum(), comb(pr[pr > 0]).sum(), comb(pp[pp > 0]).sum(), comb(n)
    exp = sa * sb / tot; ari = (sij - exp) / (0.5 * (sa + sb) - exp)
    frac_of_ref = cnt / pr[ri]; frac_of_pred = cnt / pp[pi]
    big_ref = pr >= 2000
    split = sum(1 for b in np.nonzero(big_ref)[0] if np.sum((ri == b) & (frac_of_ref >= .05)) >= 2)
    merged = sum(1 for q in np.unique(pi) if pp[q] >= 2000 and np.sum((pi == q) & (frac_of_pred >= .05)) >= 2)
    return dict(voi_split=float(h_pred_given_ref), voi_merge=float(h_ref_given_pred), voi=float(h_pred_given_ref + h_ref_given_pred),
                adjusted_rand=float(ari), reference_bands=int(big_ref.sum()), predicted_pieces=int(np.sum(pp >= 2000)),
                bands_split=int(split), pieces_merged=int(merged))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--boxes', default=BOXES); a.add_argument('--gpu', default='0')
    a.add_argument('--cutoffs', default='0.15,0.175,0.2,0.225,0.25,0.3'); a.add_argument('--aff-cuts', default='0.55,0.6,0.625,0.65,0.7'); a.add_argument('ckpts', nargs='+')
    g = a.parse_args()
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bi, bs in enumerate(g.boxes.split(';')):
        box = [int(v) for v in bs.split(',')]; o, n = np.array(box[:3]), tuple(box[3:]); nl = tuple(v // 2 for v in n)
        bd = out / ('box%d' % bi); bd.mkdir(exist_ok=True)
        if not (bd / 'ref.npz').exists():
            subprocess.run([str(UFSM), 'band', CODES, '4.8', bs, str(bd / 'band.raw')], check=True, capture_output=True)
            k = np.fromfile(bd / 'band.raw', np.uint8).reshape(nl).astype(np.int16); k[k == 255] = -1
            ct = read(ROOT, CT + '/1', (o + 1) // 2, nl, bd / 'ct.raw')   # 4.8 um CT for the inside mask
            k[ct == 0] = -1
            lab, _ = ref_instances(k)
            np.savez_compressed(bd / 'ref.npz', k=k, lab=lab, inside=ct > 0)
            (bd / 'band.raw').unlink(); (bd / 'ct.raw').unlink()
        R = np.load(bd / 'ref.npz'); ref, inside = R['lab'], R['inside']; known = (R['k'] >= 0) & (ref > 0)
        for spec in g.ckpts:
            name, ck = spec.split('=', 1)
            cd = bd / name; cd.mkdir(exist_ok=True)
            res = cd / 'bands.json'
            if res.exists(): rows += json.load(open(res)); continue
            nout = cout_of(ck); result = []
            predict(ck, box, cd / 'ch0', g.gpu, 0)
            p0 = pool2max(read(cd / 'ch0', '2.4', (0, 0, 0), n, cd / 'p.raw'))
            for cut in [float(c) for c in g.cutoffs.split(',')]:
                lab = pieces(p0 >= int(round(cut * 255)), inside)
                result.append(dict(name=name, box=box, method='recto>=%.2f' % cut, **compare(ref, lab, known)))
            if nout >= 7:
                bnd = np.zeros(nl, bool)
                for c in (1, 2, 3):
                    predict(ck, box, cd / ('ch%d' % c), g.gpu, c)
                    bnd |= pool2max(read(cd / ('ch%d' % c), '2.4', (0, 0, 0), n, cd / 'p.raw') < 128)
                lab = pieces(bnd, inside)
                result.append(dict(name=name, box=box, method='affinity d1', **compare(ref, lab, known)))
                dmax = None   # offset-8 channels: per label voxel the strongest "different" (255 - affinity) of the 2x block
                for c in (4, 5, 6):
                    predict(ck, box, cd / ('ch%d' % c), g.gpu, c)
                    dv = pool2max(255 - read(cd / ('ch%d' % c), '2.4', (0, 0, 0), n, cd / 'p.raw'))
                    dmax = dv if dmax is None else np.maximum(dmax, dv)
                for t in [float(c) for c in g.aff_cuts.split(',')]:   # boundary where "same" < t
                    result.append(dict(name=name, box=box, method='affinity d8<%.3f' % t,
                                       **compare(ref, pieces(dmax > int(round((1 - t) * 255)), inside), known)))
            (cd / 'p.raw').unlink(missing_ok=True)
            for m in ('recto', 'affinity d8'):   # best cutoff per method (tuned on this box, the same way for every checkpoint)
                c = [r for r in result if r['method'].startswith(m)]
                if c: b = min(c, key=lambda r: r['voi']); result.append(dict(b, method='BEST ' + b['method']))
            json.dump(result, open(res, 'w'), indent=1); rows += result
    for r in rows:
        print('%-16s box%s %-12s VOI %.3f (split %.3f merge %.3f) ARAND %.3f | bands %d pieces %d split %d merged %d' %
              (r['name'], r['box'][:1], r['method'], r['voi'], r['voi_split'], r['voi_merge'], r['adjusted_rand'], r['reference_bands'],
               r['predicted_pieces'], r['bands_split'], r['pieces_merged']), flush=True)
    json.dump(rows, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

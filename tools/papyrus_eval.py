#!/usr/bin/env python3
"""Evaluation of task-papyrus checkpoints (papyrus vs void) on held-out Paris4 boxes, without human labels (2026-10-11).

usage: papyrus_eval.py --out DIR NAME=CKPT [...] [--cuts 50,65,80,95,110,128] [--boxes z,y,x,nz,ny,nx;...] [--sigma 1] [--gpu 0]
       [--root R --ct G --cache C --labels L] [--window 544,16] [--blocks 2,4,4] [--png-cuts 65,80,95,110] [--crop 512]

Per box and cutoff t the reference is the training label itself: the CT smoothed by a Gaussian (sigma) above t (CT 0 = outside the
scroll mask = background). A conditioned checkpoint is predicted at every cutoff (predict --papyrus-cut t); an unconditioned one
(marginalised over U(lo, hi) in training) once, read at the probability (t - lo) / (hi - lo). Methods: the network(s), `raw` (the
unsmoothed CT above t) and `reference` (the label itself). Per method and cutoff:
  frac      papyrus fraction of the CT > 0 voxels (ref_frac: the reference's);
  iou, acc  agreement with the reference (IoU of papyrus, voxel accuracy over CT > 0);
  recto     recall of the labelled recto voxels (training-4p8 selected wraps, nearest-upsampled to 2.4 um; CT > 0) as papyrus;
  air0      fraction of the CT == 0 voxels called background (predict writes 0 there: 1 by construction);
  blk_sd    standard deviation of the papyrus fraction over sub-blocks (--blocks); blk_dev the mean |method - reference| per block.
PNGs DIR/box<i>/<name>-{z,y,x}.png: the mid planes (central --crop square), one row per --png-cuts cutoff (top = lowest);
columns CT | network overlaid | reference overlaid (papyrus tinted orange, alpha 0.45 p; the marginal net's binary map at the cutoff).
"""
import argparse, json, struct, subprocess, sys, zlib
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_grid import label_box, upsample

UFSM = Path(__file__).resolve().parent.parent / 'build' / 'ufsm'
D = Path.home() / 'ufsm-data'
ROOT = 'https://dl.ash2txt.org/community-uploads/forrest/volcomp'
CT = 'PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr'
LABELS = D / 'gt/paris4-selected-20260623/training-4p8.zarr'
BOXES = '48128,17408,16384,384,1024,1024;48512,17408,16384,512,1024,1024'


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(' '.join(map(str, cmd))[:400] + '\n' + r.stderr[-2000:])
    return r.stderr


def read(store, key, lo, size, path, cache=None):
    run([UFSM, 'read', store, key, *lo, *size, path, '--threads', '16'] + (['--cache', cache] if cache else []))
    v = np.fromfile(path, np.uint8).reshape(tuple(int(s) for s in size)); Path(path).unlink(); return v


def papyrus_header(ck):
    """the papyrus block of a checkpoint header (None if not a papyrus net)"""
    with open(ck, 'rb') as f: h = f.read(1 << 16).split(b'\n', 1)[0]
    j = json.loads(h[4:].decode())
    return j['extra'].get('papyrus') if j.get('extra', {}).get('task') == 'papyrus' else None


def smooth(ct, sigma, slab=64):
    """Gaussian-smoothed CT as float16, in z slabs with a 4 sigma halo (the same values as one call on the box, mode nearest)"""
    if sigma <= 0: return ct.astype(np.float16)
    h = int(4 * sigma + 0.5) + 1; out = np.empty(ct.shape, np.float16)
    for z0 in range(0, ct.shape[0], slab):
        a, b = max(z0 - h, 0), min(z0 + slab + h, ct.shape[0])
        out[z0:z0 + slab] = ndi.gaussian_filter(ct[a:b].astype(np.float32), sigma, mode='nearest')[z0 - a:z0 - a + min(slab, ct.shape[0] - z0)]
    return out


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                          chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def overlay(ct, p):
    """grey CT with papyrus tinted orange, alpha 0.45 p (p in 0..1)"""
    g = ct.astype(np.float32)[..., None].repeat(3, -1)
    a = 0.45 * p[..., None]
    return np.clip(g * (1 - a) + np.array([255, 120, 0], np.float32) * a, 0, 255).astype(np.uint8)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--boxes', default=BOXES); a.add_argument('--gpu', default='0')
    a.add_argument('--root', default=ROOT); a.add_argument('--ct', default=CT); a.add_argument('--cache', default=str(D / 'cache'))
    a.add_argument('--labels', default=str(LABELS)); a.add_argument('--window', default='416,16')
    a.add_argument('--cuts', default='50,65,80,95,110,128'); a.add_argument('--png-cuts', default='65,80,95,110')
    a.add_argument('--sigma', type=float, default=1.0); a.add_argument('--blocks', default='2,4,4'); a.add_argument('--crop', type=int, default=512)
    a.add_argument('--predict-args', default=''); a.add_argument('ckpts', nargs='*')
    g = a.parse_args()
    cuts = [float(c) for c in g.cuts.split(',')]; pcuts = [float(c) for c in g.png_cuts.split(',')]
    win, halo = g.window.split(','); nb = [int(v) for v in g.blocks.split(',')]
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for bi, bs in enumerate(g.boxes.split(';')):
        box = [int(v) for v in bs.split(',')]; o, n = np.array(box[:3]), np.array(box[3:])
        bd = out / ('box%d' % bi); bd.mkdir(exist_ok=True)
        ct = read(g.root, g.ct + '/0', o, n, bd / 'ct.raw', g.cache)
        valid = ct > 0
        lo, ln = label_box(o, n)
        recto = upsample(read(g.labels, '4.8', lo, ln, bd / 'lab.raw') > 0, o, n) & valid
        s = smooth(ct, g.sigma)
        bsz = [n[d] // nb[d] for d in range(3)]
        def blocks(m):
            f = []
            for i in range(nb[0]):
                for j in range(nb[1]):
                    for k in range(nb[2]):
                        sl = tuple(slice(q * bsz[d], (q + 1) * bsz[d]) for d, q in enumerate((i, j, k)))
                        v = valid[sl]; f.append(m[sl][v].mean() if v.any() else np.nan)
            return np.array(f)
        def score(name, t, y, ref, p=None):
            inter = np.count_nonzero(y & ref); uni = np.count_nonzero(y | ref)
            fb, fr = blocks(y), blocks(ref)
            r = dict(box=bi, method=name, cut=t, frac=float(np.count_nonzero(y) / np.count_nonzero(valid)),
                     ref_frac=float(np.count_nonzero(ref) / np.count_nonzero(valid)), iou=float(inter / max(uni, 1)),
                     acc=float(1 - np.count_nonzero(y ^ ref) / np.count_nonzero(valid)), recto=float(y[recto].mean()),
                     air0=float(1 - np.count_nonzero(p[~valid]) / max(np.count_nonzero(~valid), 1)) if p is not None else 1.0,
                     blk_sd=float(np.nanstd(fb)), blk_dev=float(np.nanmean(np.abs(fb - fr))))
            rows.append(r); print(json.dumps(r), flush=True)
        def ref_at(t): return (s > t) & valid   # (s > t is 0 outside the mask only up to smoothing: masked explicitly)
        for t in cuts:
            ref = ref_at(t); score('reference', t, ref, ref); score('raw', t, valid & (ct > t), ref)
        c0 = [(n[d] - min(g.crop, n[d])) // 2 for d in range(3)]
        cr = [slice(c0[d], c0[d] + min(g.crop, n[d])) for d in range(3)]
        mid = {'z': (n[0] // 2, cr[1], cr[2]), 'y': (cr[0], n[1] // 2, cr[2]), 'x': (cr[0], cr[1], n[2] // 2)}
        for spec in g.ckpts:
            name, ck = spec.split('=', 1)
            ph = papyrus_header(ck)
            if ph is None: raise SystemExit(ck + ': not a task papyrus checkpoint')
            cond = int(ph.get('cond', 0)); tlo, thi = ph['cut']
            imgs = {k: [] for k in mid}
            pm = None
            for t in cuts:
                if cond or pm is None:
                    pd = bd / ('%s-t%g' % (name, t) if cond else name)
                    if not (pd / 'zarr.json').exists():
                        run([UFSM, 'predict', ck, g.root, g.ct, pd, '--um', '2.4', '--box', bs, '--window', win, '--halo', halo, '--shard', '512',
                             '--grid-origin', '0,0,0', '--q', '0', '--threads', '16', '--levels', '1', '--gpu', g.gpu, '--cache', g.cache]
                            + (['--papyrus-cut', '%g' % t] if cond else []) + g.predict_args.split())
                    pm = read(pd, '2.4', (0, 0, 0), n, bd / 'p.raw')
                thr = 127.5 if cond else 255.0 * min(max((t - tlo) / max(thi - tlo, 1e-6), 0.0), 1.0)
                y = (pm > thr) & valid
                ref = ref_at(t)
                score(name, t, y, ref, pm)
                if t in pcuts:
                    pp = (pm.astype(np.float32) / 255.0) if cond else y.astype(np.float32)
                    for k, sl in mid.items():
                        c2 = ct[sl]
                        imgs[k].append(np.concatenate([np.repeat(c2[..., None], 3, -1), overlay(c2, pp[sl]), overlay(c2, ref[sl].astype(np.float32))], 1))
            for k, im in imgs.items():
                if im: png(bd / ('%s-%s.png' % (name, k)), np.concatenate(im, 0))
        del s
    (out / 'papyrus_eval.json').write_text(json.dumps(rows, indent=1))
    print('wrote', out / 'papyrus_eval.json')


if __name__ == '__main__':
    main()

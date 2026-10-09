#!/usr/bin/env python3
"""Score checkpoints against the m7 surface probabilities on held-out scans, at the voxel size m7 ran at (~9 um).

For each scan: two boxes of BOX^3 voxels at the m7 level (z at 1/3 and 2/3 of the scan, y/x where the m7 probability mass in
that slab is largest, from the m7 pyramid at 16x). Each checkpoint predicts the box at the same CT level (ufsm predict
--level L); predictions and m7 are compared over CT > 0 through their joint 256 x 256 histogram:
  auc      area under ROC of the prediction for m7 >= 0.5
  f1       best F1 over prediction cutoffs against m7 >= 0.5 (and the cutoff)
  mae      mean |p - p_m7|;  r  Pearson correlation of p and p_m7
usage: score_m7.py --out DIR NAME=CKPT [...] [--scrolls PHerc0172,PHerc0826,PHerc1299] [--box 512] [--gpu 0] [--window 544,16]
"""
import argparse, json, subprocess, sys, urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

UFSM = Path(__file__).resolve().parent.parent / 'build/ufsm'
ROOT = 'https://dl.ash2txt.org/community-uploads/forrest/volcomp'
INV = '/vesuvius/ufsm/runs/m7-finetune/m7_inventory.json'
AXES = '/vesuvius/ufsm/runs/m7-finetune/axes'


def read(store, key, lo, size, path):
    r = subprocess.run([str(UFSM), 'read', str(store), key, *map(str, lo), *map(str, size), str(path), '--threads', '16'], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-1500:])
    v = np.fromfile(path, np.uint8).reshape(size); Path(path).unlink(); return v


def m7_levels(m7):
    a = json.loads(urllib.request.urlopen(f'{ROOT}/{m7}/zarr.json', timeout=60).read())['attributes']['volcomp']
    return [l['path'] for l in a['levels']], a['shape']


def pick_boxes(m7, box, tmp):
    """two boxes (m7-level voxels): z at 1/3 and 2/3, y/x maximising the m7 mass of the box at that z (from the 16x level)"""
    paths, shape = m7_levels(m7)
    c = read(ROOT, f'{m7}/{paths[4]}', (0, 0, 0), [int(np.ceil(s / 16)) for s in shape], tmp / 'm.raw').astype(np.float32)
    b16 = box // 16; out = []
    for fz in (1 / 3, 2 / 3):
        z0 = int(shape[0] * fz) // 16
        slab = c[max(0, z0 - b16 // 2): z0 + b16 // 2].sum(axis=0)
        win = ndi.uniform_filter(slab, b16, mode='constant')
        win[:b16 // 2] = win[-b16 // 2:] = 0; win[:, :b16 // 2] = win[:, -b16 // 2:] = 0
        y, x = np.unravel_index(np.argmax(win), win.shape)
        out.append([int(max(0, z0 * 16 - box // 2)), int(max(0, y * 16 - box // 2)), int(max(0, x * 16 - box // 2)), box, box, box])
    return out, paths[0]


def predict(ck, r, box, out, gpu, window):
    if (out / 'zarr.json').exists(): return
    scan = r['ct'].split('/')[-1].replace('.zarr', '')
    cmd = [str(UFSM), 'predict', ck, ROOT, r['ct'], str(out), '--um', str(r['native_um']), '--level', str(r['source_level']),
           '--box', ','.join(map(str, box)), '--window', str(window[0]), '--halo', str(window[1]), '--shard', '512', '--ema', '1', '--q', '8',
           '--threads', '16', '--levels', '1', '--gpu', str(gpu), '--channel', '0', '--cache', '/vesuvius/ufsm/cache']
    ax = Path(AXES) / f'{scan}.json'
    if ax.exists(): cmd += ['--axis', str(ax)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(p.stderr[-2000:])


def metrics(h):
    """h[p, t]: joint histogram of prediction byte p and m7 byte t"""
    pos = h[:, 128:].sum(axis=1); neg = h[:, :128].sum(axis=1)
    tp = np.cumsum(pos[::-1])[::-1]; fp = np.cumsum(neg[::-1])[::-1]   # predicted positive at cutoff >= c
    P, N = pos.sum(), neg.sum()
    auc = float(np.sum(neg * (np.cumsum(pos[::-1])[::-1] - pos / 2)) / max(P * N, 1))
    f1 = 2 * tp / np.maximum(tp + fp + P, 1); c = int(np.argmax(f1))
    pv = np.arange(256)[:, None] / 255.; tv = np.arange(256)[None, :] / 255.; n = h.sum()
    mae = float((h * np.abs(pv - tv)).sum() / n)
    mp, mt = (h * pv).sum() / n, (h * tv).sum() / n
    cov = (h * (pv - mp) * (tv - mt)).sum() / n; vp = (h * (pv - mp) ** 2).sum() / n; vt = (h * (tv - mt) ** 2).sum() / n
    return dict(auc=auc, f1=float(f1[c]), cutoff=c / 255., mae=mae, r=float(cov / np.sqrt(vp * vt + 1e-12)), m7_pos=float(P / n))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--scrolls', default='PHerc0172,PHerc0826,PHerc1299'); a.add_argument('--box', type=int, default=512)
    a.add_argument('--gpu', default='0'); a.add_argument('--window', default='544,16'); a.add_argument('ckpts', nargs='+')
    g = a.parse_args()
    window = [int(v) for v in g.window.split(',')]
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    inv = [r for r in json.load(open(INV)) if r['scroll'] in g.scrolls.split(',')]
    rows = []
    for r in inv:
        scan = r['ct'].split('/')[-1][:14]; sd = out / f"{r['scroll']}-{scan}"; sd.mkdir(exist_ok=True)
        if (sd / 'boxes.json').exists(): boxes, key = json.load(open(sd / 'boxes.json'))
        else: boxes, key = pick_boxes(r['m7'], g.box, sd); json.dump([boxes, key], open(sd / 'boxes.json', 'w'))
        for bi, box in enumerate(boxes):
            bd = sd / f'box{bi}'; bd.mkdir(exist_ok=True)
            if not (bd / 'm7.npy').exists():
                m7 = read(ROOT, f"{r['m7']}/{key}", box[:3], box[3:], bd / 't.raw'); np.save(bd / 'm7.npy', m7)
                ct = read(ROOT, f"{r['ct']}/{r['source_level']}", box[:3], box[3:], bd / 'c.raw'); np.save(bd / 'inside.npy', ct > 0)
            m7 = np.load(bd / 'm7.npy'); inside = np.load(bd / 'inside.npy')
            for spec in g.ckpts:
                name, ck = spec.split('=', 1)
                res = bd / f'{name}.json'
                if res.exists(): rows.append(json.load(open(res))); continue
                predict(ck, r, box, bd / name, g.gpu, window)
                lev = sorted(p.name for p in (bd / name).iterdir() if p.is_dir())[0]
                p = read(bd / name, lev, (0, 0, 0), box[3:], bd / 'p.raw')
                h = np.bincount(p[inside].astype(np.int64) * 256 + m7[inside], minlength=65536).reshape(256, 256)
                row = dict(name=name, scroll=r['scroll'], scan=scan, um=r['m7_um'], box=box, **metrics(h))
                json.dump(row, open(res, 'w')); rows.append(row)
    for r in rows:
        print('%-14s %-11s %s %5.2f um box%s  AUC %.3f  F1 %.3f @%.2f  MAE %.3f  r %.3f  (m7 pos %.3f)' %
              (r['name'], r['scroll'], r['scan'], r['um'], r['box'][:1], r['auc'], r['f1'], r['cutoff'], r['mae'], r['r'], r['m7_pos']), flush=True)
    json.dump(rows, open(out / 'summary.json', 'w'), indent=1)
    names = sorted({r['name'] for r in rows})
    for n in names:
        rs = [r for r in rows if r['name'] == n]
        print('MEAN %-14s AUC %.3f  F1 %.3f  MAE %.3f  r %.3f' % (n, *[np.mean([r[k] for r in rs]) for k in ('auc', 'f1', 'mae', 'r')]))


if __name__ == '__main__':
    main()

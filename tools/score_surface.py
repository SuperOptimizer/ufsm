#!/usr/bin/env python3
"""Score Paris4 surface checkpoints on held-out boxes: pixel F1 / ROC AUC against the thin reference labels (CT air
excluded, best cutoff on the same box) and sheet connectivity (tools/sheet_connectivity.py: merges, splits, missed).
usage: score_surface.py --out DIR NAME=CKPT [NAME=CKPT ...] [--boxes z,y,x,nz,ny,nx;...] [--gpu 0]
Prediction settings are fixed (window 528, halo 8, shard 512, EMA, q 8) so checkpoints compare like for like."""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_connectivity import analyse, pool2, read, UFSM
from sheet_diagnostics import binary_scores

ROOT = '/vesuvius/usrm/volcomp'
CT = 'PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr'
LABELS = '/vesuvius/ufsm/gt/paris4-sheet-20261003/training-4p8.zarr'
AXIS = '/vesuvius/ufsm/runs/noisy-labels-20261005/axis-0.json'
# development ROI and the rest of the 1024^3 training-excluded holdout below it
BOXES = '48128,17408,16384,384,1024,1024;48512,17408,16384,512,1024,1024'


WINDOW = [528, 8]   # prediction window / halo (--window W,H; 6-level nets need W divisible by 32: 544,16)


def predict(ckpt, box, out, gpu):
    if (out / 'zarr.json').exists(): return
    cmd = [str(UFSM), 'predict', ckpt, ROOT, CT, str(out), '--um', '2.4', '--box', ','.join(map(str, box)), '--window', str(WINDOW[0]),
           '--halo', str(WINDOW[1]), '--shard', '512', '--grid-origin', '0,0,0', '--ema', '1', '--q', '8', '--threads', '16', '--axis', AXIS,
           '--levels', '1', '--gpu', str(gpu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-2000:])


def f1_auc(prob, box, tmp):
    o, n = np.array(box[:3]), tuple(box[3:])
    ct = read(ROOT, CT + '/0', o, n, tmp / 'ct.raw')
    # same native -> label mapping as tools/sheet_diagnostics.py (label voxel (i + 1) // 2 from (origin + 1) // 2)
    coarse = read(LABELS, '4.8', (o + 1) // 2, tuple(v // 2 + 1 for v in n), tmp / 'lab.raw') > 0
    idx = [(np.arange(v) + 1) // 2 for v in n]
    pos = np.zeros(256, np.int64); neg = pos.copy()
    for z in range(0, n[0], 16):
        t = coarse[np.ix_(idx[0][z:z + 16], idx[1], idx[2])]
        v = ct[z:z + 16] > 0; p = prob[z:z + 16]
        pos += np.bincount(p[t & v], minlength=256); neg += np.bincount(p[~t & v], minlength=256)
    return binary_scores(pos, neg), read(LABELS, '4.8', o // 2, tuple(v // 2 for v in n), tmp / 'lab2.raw') > 0


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--window', default='528,8'); a.add_argument('--boxes', default=BOXES); a.add_argument('--gpu', default='0')
    a.add_argument('ckpts', nargs='+')
    g = a.parse_args()
    WINDOW[:] = [int(v) for v in g.window.split(',')]
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    boxes = [[int(v) for v in b.split(',')] for b in g.boxes.split(';')]
    rows = []
    for spec in g.ckpts:
        name, ck = spec.split('=', 1)
        for bi, box in enumerate(boxes):
            d = out / name / ('box%d' % bi); d.mkdir(parents=True, exist_ok=True)
            res = d / 'score.json'
            if not res.exists():
                predict(ck, box, d / 'pred', g.gpu)
                prob = read(d / 'pred', '2.4', (0, 0, 0), tuple(box[3:]), d / 'prob.raw')
                s, lab = f1_auc(prob, box, d)
                conn = analyse(pool2(prob), lab, [0.3, 0.4, 0.5, s['best_cutoff']])
                json.dump(dict(name=name, ckpt=ck, box=box, pixel=s, connectivity=conn), open(res, 'w'), indent=1)
                for f in ('prob.raw', 'ct.raw', 'lab.raw', 'lab2.raw'): (d / f).unlink(missing_ok=True)
            r = json.load(open(res)); rows.append(r)
            c = r['connectivity']['results'][-1]
            print('%-16s box%d  F1 %.4f  AUC %.4f  cutoff %.2f | at best cutoff: pieces %d, merged %d (%d sheets), split %d, missed %d, off-label %.2f'
                  % (name, bi, r['pixel']['best_binary_f1'], r['pixel']['roc_auc'], r['pixel']['best_cutoff'], c['predicted_components'],
                     c['merged_components'], c['sheets_in_merges'], c['split_sheets'], c['missed_sheets'], c['orphan_fraction']), flush=True)
    json.dump(rows, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

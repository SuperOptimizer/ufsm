#!/usr/bin/env python3
"""Score checkpoints on the held-out boxes of the open-data surface labels (tools/build_open_surfaces.py), per scroll and
per test scale.

For each scan and level L (0 = native, 1 = 2x coarser, ...): predict the whole held-out box at level L (ufsm predict
--level L, window 544/16) and compare with the binary labels at that level (finer than the label level: nearest
upsampling; coarser: the any-positive label pyramid). Labels cover only some sheets, so the scored voxels are CT > 0
within TRUST_UM of a labelled surface (the training trust band); 'all' scores every CT voxel for reference.
Metrics: ROC AUC and best F1 (tools/sheet_diagnostics.binary_scores).
usage: score_open.py --out DIR NAME=CKPT [...] [--scans PHerc0139,...] [--levels 0,1,2] [--gpu 0]
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheet_diagnostics import binary_scores

UFSM = Path(__file__).resolve().parent.parent / 'build/ufsm'
GT = Path('/vesuvius/ufsm/gt/open-surfaces-20261007')
TRUST_UM = 48.0


def read(store, key, lo, size, path):
    r = subprocess.run([str(UFSM), 'read', str(store), key, *map(str, lo), *map(str, size), str(path), '--threads', '16'], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-1500:])
    v = np.fromfile(path, np.uint8).reshape(size); Path(path).unlink(); return v


def predict(ck, b, L, box, out, gpu):
    if (out / 'zarr.json').exists(): return
    cmd = [str(UFSM), 'predict', ck, b['root'], b['ct'], str(out), '--um', str(b['um']), '--level', str(L), '--box', ','.join(map(str, box)),
           '--window', '544', '--halo', '16', '--shard', '512', '--ema', '1', '--q', '8', '--threads', '16', '--levels', '1', '--gpu', str(gpu),
           '--channel', '0', '--cache', '/vesuvius/ufsm/cache']
    if b.get('axis'): cmd += ['--axis', b['axis']]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-2000:])


def labels_at(b, L, box, tmp):
    """binary labels on the level-L grid of box (level-L voxels)"""
    keys = sorted([p.name for p in Path(b['labels']).iterdir() if p.is_dir()], key=float)
    ll = b['label_level']
    if L >= ll: return read(b['labels'], keys[L - ll], box[:3], box[3:], tmp) > 0
    f = 2 ** (ll - L); o = np.array(box[:3]); n = box[3:]
    lab = read(b['labels'], keys[0], o // f, [(o[i] + n[i] - 1) // f - o[i] // f + 1 for i in range(3)], tmp) > 0
    idx = [(o[i] + np.arange(n[i])) // f - o[i] // f for i in range(3)]
    return lab[np.ix_(*idx)]


def near_label(lab, r, slab=64):
    """EDT(~lab) <= r, computed in z slabs with an r-plane halo (exact for distances <= r, bounded memory)"""
    out = np.zeros_like(lab); h = int(np.ceil(r)) + 1
    for z in range(0, lab.shape[0], slab):
        a, b = max(0, z - h), min(lab.shape[0], z + slab + h)
        sub = lab[a:b]
        if not sub.any(): continue
        out[z:z + slab] = (ndi.distance_transform_edt(~sub) <= r)[z - a: z - a + min(slab, lab.shape[0] - z)]
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--scans', default=''); a.add_argument('--levels', default='0,1,2')
    a.add_argument('--gpu', default='0'); a.add_argument('--gt', default=str(GT), help='label build directory (scans.json)'); a.add_argument('ckpts', nargs='+')
    g = a.parse_args()
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    builds = {k: v for k, v in json.load(open(Path(g.gt) / 'scans.json')).items() if 'error' not in v and (not g.scans or k in g.scans.split(','))}
    rows = []
    for scroll, b in sorted(builds.items()):
        for L in map(int, g.levels.split(',')):
            f = 2 ** L; box = [v // f for v in b['holdout']]
            if min(box[3:]) < 64: continue
            d = out / scroll / f'L{L}'; d.mkdir(parents=True, exist_ok=True)
            if not (d / 'mask.npy').exists():
                lab = labels_at(b, L, box, d / 'l.raw')
                ct = read(b['root'], f"{b['ct']}/{L}", box[:3], box[3:], d / 'c.raw') > 0
                near = near_label(lab, max(1.0, TRUST_UM / (b['um'] * f)))
                np.save(d / 'lab.npy', lab); np.save(d / 'mask.npy', near & ct); np.save(d / 'ct.npy', ct)
            lab, near, ct = np.load(d / 'lab.npy'), np.load(d / 'mask.npy'), np.load(d / 'ct.npy')
            for spec in g.ckpts:
                name, ck = spec.split('=', 1); res = d / f'{name}.json'
                if not res.exists():
                    predict(ck, b, L, box, d / name, g.gpu)
                    key = sorted(p.name for p in (d / name).iterdir() if p.is_dir())[0]
                    p = read(d / name, key, (0, 0, 0), box[3:], d / 'p.raw')
                    r = dict(name=name, scroll=scroll, level=L, um=round(b['um'] * f, 3), box=box)
                    for tag, m in (('near', near), ('all', ct)):
                        pos = np.bincount(p[lab & m], minlength=256); neg = np.bincount(p[~lab & m], minlength=256)
                        s = binary_scores(pos, neg); r[tag] = dict(auc=s['roc_auc'], f1=s['best_binary_f1'], cutoff=s['best_cutoff'], pos_frac=float(pos.sum() / max(1, pos.sum() + neg.sum())))
                    json.dump(r, open(res, 'w'), indent=1)
                r = json.load(open(res)); rows.append(r)
                print('%-12s %-11s L%d %6.2f um  near: AUC %.3f F1 %.3f @%.2f | all: AUC %.3f F1 %.3f' % (
                    name, scroll, L, r['um'], r['near']['auc'] or 0, r['near']['f1'], r['near']['cutoff'], r['all']['auc'] or 0, r['all']['f1']), flush=True)
    json.dump(rows, open(out / 'summary.json', 'w'), indent=1)
    for n in sorted({r['name'] for r in rows}):
        for L in sorted({r['level'] for r in rows}):
            rs = [r for r in rows if r['name'] == n and r['level'] == L]
            if rs: print('MEAN %-12s L%d  near AUC %.3f F1 %.3f  (%d scans)' % (n, L, np.mean([r['near']['auc'] or 0 for r in rs]), np.mean([r['near']['f1'] for r in rs]), len(rs)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Sheet-level connectivity of a surface prediction against a binary reference (merges, splits, missed sheets).

Pixel F1 does not say whether two wraps were fused into one predicted sheet. This script works at the 4.8 um label
level: the prediction is max-pooled 2x, thresholded, and both volumes are split into 26-connected components.
Reference components stand in for sheets (where the reference itself touches, its components are already merged;
that count is reported). For each cutoff:
  merges   predicted components that cover >= MIN_COVER voxels of two or more reference sheets
  splits   reference sheets whose covered voxels fall into two or more predicted components (each >= MIN_COVER)
  missed   reference sheets with recall below 0.1
  orphans  predicted voxels farther than TOL label voxels from any reference voxel (fraction)
usage: sheet_connectivity.py --pred PRED_STORE --box z,y,x,nz,ny,nx [--labels STORE] [--thr 0.3,0.4,0.5] [--out x.json]
PRED_STORE is a `ufsm predict` output whose level-0 array starts at the box origin (key 2.4).
"""
import argparse, json, subprocess, tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

UFSM = Path(__file__).resolve().parent.parent / 'build/ufsm'
LABELS = '/vesuvius/ufsm/gt/paris4-sheet-20261003/training-4p8.zarr'
MIN_SHEET, MIN_COVER, TOL = 200, 50, 2   # label voxels


def read(store, key, lo, size, path):
    r = subprocess.run([str(UFSM), 'read', str(store), key, *map(str, lo), *map(str, size), str(path), '--threads', '8'],
                       capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    return np.fromfile(path, np.uint8).reshape(size)


def pool2(v):   # max over 2x2x2 blocks (odd tails dropped)
    s = tuple(n // 2 * 2 for n in v.shape); v = v[:s[0], :s[1], :s[2]]
    return v.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).max(axis=(1, 3, 5))


def analyse(prob, ref, thr):
    full = np.ones((3, 3, 3), bool)
    rl, nr = ndi.label(ref, full)
    rsz = np.bincount(rl.ravel(), minlength=nr + 1); sheets = np.nonzero(rsz >= MIN_SHEET)[0]; sheets = sheets[sheets > 0]
    near = ndi.binary_dilation(ref, full, iterations=TOL)
    # nearest reference component for every voxel within TOL (so a prediction one voxel off the thin label still counts)
    _, idx = ndi.distance_transform_edt(rl == 0, return_indices=True)
    owner = rl[tuple(idx)]; owner[~near] = 0
    out = []
    for t in thr:
        p = prob >= int(round(t * 255))
        pl, npc = ndi.label(p, full)
        # (pred, ref) co-occurrence over predicted voxels
        pair = pl.astype(np.int64) * (nr + 1) + owner
        cnt = np.bincount(pair[p].ravel(), minlength=(npc + 1) * (nr + 1)).reshape(npc + 1, nr + 1)
        cover = cnt[:, sheets] >= MIN_COVER
        merges = int((cover[1:].sum(1) >= 2).sum())
        fused = int(cover[1:][cover[1:].sum(1) >= 2].sum())   # sheets involved in some merge
        splits = int((cover[1:].sum(0) >= 2).sum())
        rec = np.array([(p & (rl == s)).sum() / rsz[s] for s in sheets]) if len(sheets) else np.zeros(0)
        out.append(dict(cutoff=t, predicted_components=int(npc), merged_components=merges, sheets_in_merges=fused,
                        split_sheets=splits, missed_sheets=int((rec < 0.1).sum()), median_sheet_recall=float(np.median(rec)) if len(rec) else None,
                        orphan_fraction=float((p & ~near).sum() / max(1, p.sum())), predicted_voxels=int(p.sum())))
    return dict(reference_sheets=int(len(sheets)), reference_components=int(nr), results=out)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--pred', required=True); a.add_argument('--box', required=True)
    a.add_argument('--labels', default=LABELS); a.add_argument('--thr', default='0.3,0.4,0.5'); a.add_argument('--out')
    g = a.parse_args()
    box = [int(x) for x in g.box.split(',')]; o, n = np.array(box[:3]), tuple(box[3:])
    if any(v % 2 for v in (*o, *n)): raise SystemExit('box origin and size must be even (2x pooling onto the 4.8 um grid)')
    with tempfile.TemporaryDirectory() as d:
        prob = pool2(read(g.pred, '2.4', (0, 0, 0), n, Path(d) / 'p.raw'))
        ref = read(g.labels, '4.8', o // 2, tuple(v // 2 for v in n), Path(d) / 'l.raw') > 0
    r = analyse(prob, ref, [float(t) for t in g.thr.split(',')])
    r.update(pred=g.pred, box=box, labels=g.labels, tolerance_label_voxels=TOL, min_cover=MIN_COVER, min_sheet=MIN_SHEET)
    s = json.dumps(r, indent=1); print(s)
    if g.out: Path(g.out).write_text(s)


if __name__ == '__main__':
    main()

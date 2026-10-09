#!/usr/bin/env python3
"""tools/label_grid.py and the measurements of tools/score_sheets.py on synthetic volumes: the trainer's label mapping,
radial ridge suppression, tolerance F1 and radial profile width/offset of a Gaussian ridge on a cylinder shell."""
import json, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from label_grid import label_index, label_box, pool_max, pool_mean, upsample
import score_sheets as ss


def check(c, msg):
    if not c: raise SystemExit('FAIL: ' + msg)


def test_grid():
    check(list(label_index([7, 8, 9, 10])) == [4, 4, 5, 5], 'label j covers natives 2j-1, 2j')
    check(list(label_index([7, 8, 9, 10], legacy=True)) == [3, 4, 4, 5], 'legacy pooling is floor')
    o = np.array([8, 10, 12]); n = (6, 8, 4)
    v = np.zeros(n, np.uint8); v[1, 2, 0] = 9   # native (9, 12, 12) -> label (5, 6, 6) -> relative (1, 1, 0)
    lo, ln = label_box(o, n); check(tuple(lo) == (4, 5, 6) and tuple(ln) == (3, 4, 2), 'label box')
    pm = pool_max(v, o); check(pm.shape == (3, 4, 2) and pm[1, 1, 0] == 9 and pm.sum() == 9, 'pool_max placement')
    pl = pool_max(v, o, legacy=True); check(pl[0, 1, 0] == 9, 'legacy placement')
    m = pool_mean(np.full(n, 4, np.uint8), o); check(np.allclose(m, 4), 'pool_mean')
    lab = np.arange(np.prod(ln)).reshape(tuple(ln)); up = upsample(lab, o, n)
    check(up[1, 2, 0] == lab[1, 1, 0] and up[0, 0, 0] == lab[0, 0, 0], 'upsample is the inverse mapping')


def test_ridge():
    """cylinder shell around a vertical axis at (y, x) = (100, 100), radius 50, Gaussian ridge sigma 3"""
    o = np.array([0, 2, 4]); n = (8, 196, 192); R, sig = 50.0, 3.0
    with tempfile.TemporaryDirectory() as d:
        ax = Path(d) / 'axis.json'
        ax.write_text(json.dumps({'control_points': [{'z': -10, 'y': 100, 'x': 100}, {'z': 100, 'y': 100, 'x': 100}]}))
        axis = ss.axis_of(ax)
    z, y, x = np.meshgrid(np.arange(n[0]), np.arange(n[1]) + o[1], np.arange(n[2]) + o[2], indexing='ij')
    r = np.hypot(y - 100.0, x - 100.0)
    p = np.clip(np.rint(255 * np.exp(-0.5 * ((r - R) / sig) ** 2)), 0, 255).astype(np.uint8)
    thin = ss.radial_nms(p, o, axis)
    on = thin > 128
    check(on.any(), 'ridge survives')
    check(np.abs(r[on] - R).max() <= 1.0, 'ridge voxels lie on the surface (max |r - R| %.2f)' % np.abs(r[on] - R).max())
    check((p[4] > 128).sum() > 2.5 * on[4].sum(), 'ridge is thinner than the band')
    # reference: label voxels whose native centre is within half a label voxel of the surface
    lo, ln = label_box(o, n)
    jz, jy, jx = np.meshgrid(np.arange(ln[0]) + lo[0], np.arange(ln[1]) + lo[1], np.arange(ln[2]) + lo[2], indexing='ij')
    gt = np.abs(np.hypot(2.0 * jy - 100, 2.0 * jx - 100) - R) < 1.0
    valid = np.ones(gt.shape, bool)
    s = ss.tolerance_scores(pool_max(thin, o), gt, valid)
    check(s['tol1']['f1'] > 0.95 and s['tol2']['f1'] >= s['tol1']['f1'] >= s['tol0']['f1'], 'tolerance F1 %s' % s)
    check(s['tol0']['f1'] > 0.7, 'exact-match F1 of the ridge %.3f' % s['tol0']['f1'])
    pf = ss.profiles(p, o, gt, valid, axis)
    fw = 2 * np.sqrt(2 * np.log(2)) * sig
    check(abs(pf['fwhm_median'] - fw) < 1.0, 'FWHM %.2f vs %.2f' % (pf['fwhm_median'], fw))
    check(abs(pf['offset_mean']) < 0.6 and pf['double_peak'] < 0.01, 'offset %.2f, double %.3f' % (pf['offset_mean'], pf['double_peak']))
    # a second shell 9 voxels outside gives double peaks
    p2 = np.maximum(p, np.clip(np.rint(255 * np.exp(-0.5 * ((r - R - 9) / sig) ** 2)), 0, 255).astype(np.uint8))
    check(ss.profiles(p2, o, gt, valid, axis)['double_peak'] > 0.9, 'double peaks detected')


def test_best_f1():
    tp = np.zeros(256); al = np.zeros(256); rc = np.zeros(256)
    tp[200] = 8; al[200] = 10; al[50] = 30; rc[200] = 8
    b = ss.best_f1(tp, al, rc, 10)
    check(abs(b['f1'] - 0.8) < 1e-9 and 50 / 255 < b['cutoff'] <= 200 / 255, 'best cutoff %s' % b)


if __name__ == '__main__':
    test_grid(); test_best_f1(); test_ridge()
    print('test_score_sheets: ok')

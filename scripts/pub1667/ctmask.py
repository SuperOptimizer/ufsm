#!/usr/bin/env python3
"""surfcmp.py's published vertices split by the CT under them (masked-out / air vs papyrus) and coverage recomputed per class.

  ctmask.py PUB_TIFXYZ CMP_DIR CT.raw Z,Y,X --level L [--thr 1]

CT.raw: a whole uint8 CT level (ufsm read ... 0 0 0 Z Y X), level L of the level-0 grid. A vertex counts as "on CT" when the max of
the 3^3 level-L neighbourhood is >= --thr (the masked volume is 0 outside its mask). Writes CMP_DIR/ctmask.json and adds the
column on_ct to a copy of pub_turns.tsv (pub_turns_ct.tsv).
"""
import json, sys
import numpy as np
import tifffile
from scipy.ndimage import maximum_filter

def main():
    pub, cmp_dir, raw, shp = sys.argv[1:5]
    a = sys.argv; L = int(a[a.index('--level') + 1]); thr = int(a[a.index('--thr') + 1]) if '--thr' in a else 1
    um = 2.399
    shape = tuple(int(v) for v in shp.split(','))
    ct = np.fromfile(raw, np.uint8).reshape(shape)
    ctm = maximum_filter(ct, size=3)
    r = np.load(f'{cmp_dir}/pub_dist.npz'); vi = r['vi']; d = r['d']; A = r['area']; T = r['T']; off = r['off']
    f = 2 ** L
    idx = []
    for c in ('z', 'y', 'x'):
        v = tifffile.imread(f'{pub}/{c}.tif').ravel()[vi]
        idx.append(np.clip(np.round((v + 0.5) / f - 0.5).astype(np.int64), 0, shape[len(idx)] - 1))
    on = ctm[idx[0], idx[1], idx[2]] >= thr
    out = {'on_ct_area_share': float(A[on].sum() / A.sum())}
    for nm, m in (('on_ct', on), ('off_ct', ~on)):
        for th in (50, 100, 200):
            out[f'{nm}_cov_{th}'] = float(A[m & (d < th)].sum() / max(A[m].sum(), 1e-9))
        out[f'{nm}_area_cm2'] = float(A[m].sum() * um * um * 1e-8)
        out[f'{nm}_d_med'] = float(np.median(d[m])) if m.any() else None
        mm = m & (d < 100); v, c = np.unique(off[mm], return_counts=True)
        out[f'{nm}_same_sheet'] = float(c[v == 0].sum() / max(c.sum(), 1))
    json.dump(out, open(f'{cmp_dir}/ctmask.json', 'w'), indent=1)
    for k, v in out.items(): print(f'  {k}: {v}')
    kt = np.floor(T).astype(int)
    lines = open(f'{cmp_dir}/pub_turns.tsv').read().splitlines()
    with open(f'{cmp_dir}/pub_turns_ct.tsv', 'w') as fo:
        fo.write(lines[0] + '\ton_ct\tcov100_on_ct\n')
        for ln in lines[1:]:
            k = int(ln.split('\t')[0]); m = kt == k; mo = m & on
            fo.write(f'{ln}\t{A[mo].sum() / A[m].sum():.3f}\t{A[mo & (d < 100)].sum() / max(A[mo].sum(), 1e-9):.3f}\n')

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""scripts/inkgt/mkcrops.py SEGS.TSV WIN.TSV OUT: ink eval crops (the scripts/univ/ink_prep.py layout) from label windows (2026-11-11).

SEGS.TSV rows: scroll  name  tifxyz-dir  labels  mask  ct-key  um (labels / mask: .tif or a zarr v3 2-D array dir); WIN.TSV rows: name  c0  r0  n
(label px; canvas px = label px, the open-data / HF convention). Each window -> OUT/<scroll>_<short>_<k>/: the tifxyz sub-grid (level-0 voxels,
meta.json with the crop's bbox), labels.tif, mask.tif (0/255), crop.json (segment, grid offsets, supervised px, ink share, box_l0 = the
surface's level-0 box +- 32 voxels: the held-out region), and OUT/crops.tsv (+ scroll and ct columns).
"""
import json, os, sys
import numpy as np, tifffile, zarr

def rd(p, r0, c0, n):
    if p.endswith(('.tif', '.tiff')):
        a = tifffile.imread(p)
        if a.ndim == 3: a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
        a = a[r0:r0 + n, c0:c0 + n]
    else: a = zarr.open_array(p, mode='r')[r0:r0 + n, c0:c0 + n]
    a = np.ascontiguousarray(np.asarray(a, np.uint8))
    return a * 255 if a.max() == 1 else a

def main():
    segs = {l.split('\t')[1]: l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.strip()}
    out = sys.argv[3]; os.makedirs(out, exist_ok=True); rows = []; k = {}
    for l in open(sys.argv[2]):
        f = l.rstrip('\n').split('\t')
        if len(f) < 4 or f[0] not in segs: continue
        sc, nm, tx, lab, msk, key, um = segs[f[0]]; c0, r0, n = int(f[1]), int(f[2]), int(f[3])
        meta = json.load(open(os.path.join(tx, 'meta.json'))); sc_ = float(meta['scale'][0]); g = int(round(1 / sc_))
        c0, r0, n = c0 - c0 % g, r0 - r0 % g, n - n % g; gr, gc, gn = r0 // g, c0 // g, n // g
        X, Y, Z = (tifffile.imread(os.path.join(tx, a + '.tif')) for a in 'xyz')
        xs, ys, zs = (np.ascontiguousarray(A[gr:gr + gn, gc:gc + gn]) for A in (X, Y, Z))
        L, M = rd(lab, r0, c0, n), rd(msk, r0, c0, n)
        short = nm.split('-', 1)[-1].split('_')[0] if '-' in nm else nm[:14]
        k[short] = k.get(short, -1) + 1; o = os.path.join(out, '%s_%s_%d' % (sc, short, k[short])); os.makedirs(o, exist_ok=True)
        for a, v in zip('xyz', (xs, ys, zs)): tifffile.imwrite(os.path.join(o, a + '.tif'), v)
        m = zs >= 0
        lo = [max(0, int(np.floor(v[m].min())) - 32) for v in (zs, ys, xs)]; lo = [a - a % 4 for a in lo]
        hi = [int(np.ceil(v[m].max())) + 32 for v in (zs, ys, xs)]
        mm = dict(meta); mm['bbox'] = [[float(xs[m].min()), float(ys[m].min()), float(zs[m].min())], [float(xs[m].max()), float(ys[m].max()), float(zs[m].max())]]
        json.dump(mm, open(os.path.join(o, 'meta.json'), 'w'))
        tifffile.imwrite(os.path.join(o, 'labels.tif'), L); tifffile.imwrite(os.path.join(o, 'mask.tif'), M)
        sup = M >= 128; s = int(sup.sum()); fi = float(((L >= 128) & sup).sum()) / max(1, s)
        box = lo + [hi[i] - lo[i] for i in range(3)]
        json.dump({'segment': nm, 'scroll': sc, 'ct': key, 'um': float(um), 'grid_r0': gr, 'grid_c0': gc, 'grid_n': gn, 'label_px_per_grid': g,
                   'label_c0': c0, 'label_r0': r0, 'supervised_px': s, 'ink_share': round(fi, 4), 'box_l0': box}, open(os.path.join(o, 'crop.json'), 'w'))
        rows.append('\t'.join([os.path.basename(o), nm, str(s), '%.3f' % fi, ','.join(map(str, box)), sc, key]))
    open(os.path.join(out, 'crops.tsv'), 'w').write('# crop\tsegment\tsupervised_px\tink_share\tbox_l0 (z,y,x,nz,ny,nx)\tscroll\tct\n' + '\n'.join(rows) + '\n')
    print('\n'.join(rows))

if __name__ == '__main__': main()

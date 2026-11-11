#!/usr/bin/env python3
"""scripts/univ/ink_prep.py HFINK OUT [N_PER_SEG] [CROP_PX]: the ink column's eval crops (universal-net eval harness, 2026-11-11).

From the human Paris 4 ink label segments (HF ink/phercparis4 on the 2.4 um scan, fetched by tools/pbr/hfget.sh into HFINK/wXX_*: tifxyz
x/y/z.tif + meta.json + *_inklabels.tif + *_supervision_mask.tif) pick up to N_PER_SEG windows of CROP_PX x CROP_PX label pixels per segment
with the most supervised pixels that hold both classes (ink share 5..95 %), non-overlapping, and cut each into OUT/<seg>_<k>/: the tifxyz
sub-grid (level-0 voxel coords, meta.json with the same scale and the crop's bbox), labels.tif, mask.tif, crop.json (grid / label offsets,
supervised px, ink share, the level-0 box z,y,x,nz,ny,nx of the surface +- 32 voxels for the ink prediction). Label px = canvas px =
tifxyz grid / scale (F = 1 as ink-score infers it). Deterministic: same inputs -> same crops.
"""
import json, os, sys
import numpy as np, tifffile

def main():
    src, out = sys.argv[1], sys.argv[2]; nper = int(sys.argv[3]) if len(sys.argv) > 3 else 2; cpx = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    os.makedirs(out, exist_ok=True); rows = []
    for seg in sorted(os.listdir(src)):
        d = os.path.join(src, seg)
        if not os.path.exists(os.path.join(d, 'x.tif')): continue
        meta = json.load(open(os.path.join(d, 'meta.json'))); sc = float(meta['scale'][0]); step = 1 / sc
        X, Y, Z = (tifffile.imread(os.path.join(d, a + '.tif')) for a in 'xyz')
        lab = [f for f in os.listdir(d) if f.endswith(('_inklabels.tif', '_inklabels.tiff')) and not f[0].isdigit()][0]
        msk = [f for f in os.listdir(d) if 'supervision_mask.tif' in f and not f[0].isdigit()][0]
        L = tifffile.memmap(os.path.join(d, lab)) if tifffile.TiffFile(os.path.join(d, lab)).pages[0].is_memmappable else tifffile.imread(os.path.join(d, lab))
        Mk = tifffile.memmap(os.path.join(d, msk)) if tifffile.TiffFile(os.path.join(d, msk)).pages[0].is_memmappable else tifffile.imread(os.path.join(d, msk))
        if L.ndim == 3: L = L[..., 0]
        if Mk.ndim == 3: Mk = Mk[..., 0]
        g = max(1, int(round(step)))   # label px per grid cell
        gh, gw = min(L.shape[0] // g, Z.shape[0]), min(L.shape[1] // g, Z.shape[1])
        sup = (np.asarray(Mk[:gh * g:g, :gw * g:g]) >= 128) & (Z[:gh, :gw] >= 0)
        ink = (np.asarray(L[:gh * g:g, :gw * g:g]) >= 128) & sup
        cg = cpx // g; picks = []
        S = np.pad(np.cumsum(np.cumsum(sup, 0), 1), ((1, 0), (1, 0))); I = np.pad(np.cumsum(np.cumsum(ink, 0), 1), ((1, 0), (1, 0)))
        cand = []
        for r in range(0, gh - cg, cg // 4):
            for c in range(0, gw - cg, cg // 4):
                s = S[r + cg, c + cg] - S[r, c + cg] - S[r + cg, c] + S[r, c]
                if s < 0.5 * cg * cg: continue
                i = I[r + cg, c + cg] - I[r, c + cg] - I[r + cg, c] + I[r, c]
                if 0.05 < i / s < 0.95: cand.append((s, r, c, i / s))
        for s, r, c, fi in sorted(cand, reverse=True):
            if len(picks) >= nper: break
            if any(abs(r - pr) < cg and abs(c - pc) < cg for _, pr, pc, _ in picks): continue
            picks.append((s, r, c, fi))
        for k, (s, r, c, fi) in enumerate(picks):
            o = os.path.join(out, '%s_%d' % (seg.split('_')[0], k)); os.makedirs(o, exist_ok=True)
            xs, ys, zs = X[r:r + cg, c:c + cg], Y[r:r + cg, c:c + cg], Z[r:r + cg, c:c + cg]
            for a, v in zip('xyz', (xs, ys, zs)): tifffile.imwrite(os.path.join(o, a + '.tif'), np.ascontiguousarray(v))
            m = (zs >= 0); lo = [int(np.floor(v[m].min())) - 32 for v in (zs, ys, xs)]; hi = [int(np.ceil(v[m].max())) + 32 for v in (zs, ys, xs)]
            lo = [max(0, a - a % 4) for a in lo]
            mm = dict(meta); mm['bbox'] = [[float(xs[m].min()), float(ys[m].min()), float(zs[m].min())], [float(xs[m].max()), float(ys[m].max()), float(zs[m].max())]]
            json.dump(mm, open(os.path.join(o, 'meta.json'), 'w'))
            tifffile.imwrite(os.path.join(o, 'labels.tif'), np.ascontiguousarray(np.asarray(L[r * g:(r + cg) * g, c * g:(c + cg) * g])))
            tifffile.imwrite(os.path.join(o, 'mask.tif'), np.ascontiguousarray(np.asarray(Mk[r * g:(r + cg) * g, c * g:(c + cg) * g])))
            box = lo + [hi[i] - lo[i] for i in range(3)]
            json.dump({'segment': seg, 'grid_r0': r, 'grid_c0': c, 'grid_n': cg, 'label_px_per_grid': g, 'supervised_px': int(s) * g * g,
                       'ink_share': round(float(fi), 4), 'box_l0': box}, open(os.path.join(o, 'crop.json'), 'w'))
            rows.append('%s\t%s\t%d\t%.3f\t%s' % (os.path.basename(o), seg, int(s) * g * g, fi, ','.join(map(str, box))))
    open(os.path.join(out, 'crops.tsv'), 'w').write('# crop\tsegment\tsupervised_px\tink_share\tbox_l0 (z,y,x,nz,ny,nx)\n' + '\n'.join(rows) + '\n')
    print('\n'.join(rows))

if __name__ == '__main__':
    main()

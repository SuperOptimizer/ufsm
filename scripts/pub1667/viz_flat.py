#!/usr/bin/env python3
"""Maps along the published surface's own (straightened) grid from surfcmp.py's pub_dist.npz.

  viz_flat.py CMP_DIR OUT_DIR [--rows 4] [--block 2] [--ref-id-offset N]

dist.png: distance to the nearest of our wraps (um, 0 .. 300, one hue light -> dark; > 300 darkest), invalid white.
gauge.png: which of our wraps lies on the published sheet: id - (published sheet index + offset) per vertex within 100 um;
same sheet (the scroll-wide mode) blue, one outward orange, one inward aqua, two or more grey, no wrap within 100 um light grey.
The strip is cut into --rows pieces stacked top to bottom (column order = along the published unwrap, left to right, then next row).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

def opt(k, d):
    a = sys.argv
    return a[a.index(k) + 1] if k in a else d

def main():
    cmp_dir, out = sys.argv[1:3]; rows = int(opt('--rows', '4')); blk = int(opt('--block', '2'))
    os.makedirs(out, exist_ok=True)
    r = np.load(f'{cmp_dir}/pub_dist.npz'); H, W = (int(v) for v in r['shape']); vi = r['vi']
    d = np.full(H * W, np.nan, np.float32); d[vi] = r['d']; d = d.reshape(H, W)
    g = np.full(H * W, -128, np.int16); g[vi] = r['gx'] if 'gx' in r else r['off']; g = g.reshape(H, W)
    Hb, Wb = H // blk, W // blk
    col = r['vi'] % W; T = r['T']; outer_left = np.corrcoef(col[::101], T[::101])[0, 1] < 0
    end = 'outer end (last turn) at top-left, inner end at bottom-right' if outer_left else 'inner end at top-left, outer end at bottom-right'
    def pool_mean(a):
        a = a[:Hb * blk, :Wb * blk].reshape(Hb, blk, Wb, blk)
        return np.nanmean(a, axis=(1, 3))
    import warnings; warnings.simplefilter('ignore')
    dp = pool_mean(d)
    gp = g[:Hb * blk:blk, :Wb * blk:blk]                    # nearest (categorical)
    dn = d[:Hb * blk:blk, :Wb * blk:blk]
    # dist colours: one hue (blue) light -> dark over 0..300 um
    lo = np.array([0xe8, 0xf1, 0xfb], float); hi = np.array([0x0b, 0x2a, 0x5c], float)
    t = np.clip(dp / 300.0, 0, 1)[..., None]
    rgb = (lo + (hi - lo) * t)
    rgb[np.isnan(dp)] = 255
    rgb[dp > 300] = [0x05, 0x10, 0x26]
    cat = np.full((Hb, Wb, 3), 255, float)
    valid = ~np.isnan(dn)
    cat[valid & (dn >= 100)] = [0xd9, 0xd8, 0xd4]
    m = valid & (dn < 100)
    cat[m & (gp == 0)] = [0x2a, 0x78, 0xd6]; cat[m & (gp == 1)] = [0xeb, 0x68, 0x34]; cat[m & (gp == -1)] = [0x1b, 0xaf, 0x7a]
    cat[m & (np.abs(gp) >= 2)] = [0x52, 0x51, 0x4e]
    for nm, img, legend in (('dist', rgb, [('0 um', lo), ('150 um', (lo + hi) / 2), ('300 um', hi), ('> 300 um', [5, 16, 38])]),
                            ('gauge', cat, [('same sheet', [0x2a, 0x78, 0xd6]), ('our id +1', [0xeb, 0x68, 0x34]),
                                            ('our id -1', [0x1b, 0xaf, 0x7a]), ('off by >= 2', [0x52, 0x51, 0x4e]),
                                            ('no wrap within 100 um', [0xd9, 0xd8, 0xd4])])):
        seg = (Wb + rows - 1) // rows; gap = 12; lh = 40
        canvas = np.full((rows * (Hb + gap) + lh, seg, 3), 255, np.uint8)
        for k in range(rows):
            piece = img[:, k * seg:(k + 1) * seg].astype(np.uint8)
            canvas[lh + k * (Hb + gap): lh + k * (Hb + gap) + Hb, :piece.shape[1]] = piece
        im = Image.fromarray(canvas); dr = ImageDraw.Draw(im); x = 8
        for txt, col in legend:
            dr.rectangle([x, 10, x + 20, 30], fill=tuple(int(c) for c in col), outline=(80, 80, 80)); dr.text((x + 26, 14), txt, fill=(20, 20, 20)); x += 40 + 8 * len(txt)
        dr.text((x + 20, 14), f'published unwrap grid ({H}x{W}, 1 px = {blk} grid px), {rows} rows left->right, {end}', fill=(20, 20, 20))
        im.save(f'{out}/{nm}.png', optimize=True)
        print(f'{out}/{nm}.png', im.size)

if __name__ == '__main__':
    sys.exit(main())

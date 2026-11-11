#!/usr/bin/env python3
"""scripts/inkgt/windows.py SEGS.TSV [W]: the label windows (W x W label px, default 16384) of every segment that hold supervised pixels
(the ink-splat --crop list; a whole canvas can exceed the splatter's 4 G px window). Prints  name  c0  r0  w  h  supervised_px."""
import sys, numpy as np, tifffile, zarr
G = 16
def rd(p):
    if p.endswith(('.tif', '.tiff')):
        a = tifffile.imread(p)
        if a.ndim == 3: a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
        return np.asarray(a[::G, ::G]), a.shape[:2]
    z = zarr.open_array(p, mode='r'); H, W = z.shape; o = np.empty(((H + G - 1) // G, (W + G - 1) // G), np.uint8)
    for r in range(0, H, 4096): b = z[r:min(H, r + 4096), :]; o[r // G:r // G + (b.shape[0] + G - 1) // G] = b[::G, ::G]
    return o, (H, W)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
for l in open(sys.argv[1]):
    sc, nm, tx, lab, msk, key, um = l.rstrip('\n').split('\t')
    m, (H, Wd) = rd(msk); s = m >= (128 if m.max() > 1 else 1)
    for r0 in range(0, H, W):
        for c0 in range(0, Wd, W):
            n = int(s[r0 // G:(r0 + W) // G, c0 // G:(c0 + W) // G].sum()) * G * G
            if n: print(nm, c0, r0, min(W, Wd - c0), min(W, H - r0), n, sep='\t', flush=True)

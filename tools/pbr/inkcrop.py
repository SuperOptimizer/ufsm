# inkcrop.py SEGDIR Y X N OUT.png: crop of the published ink prediction (uint8)
import sys, numpy as np, tifffile
from PIL import Image
d, y0, x0, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
t = tifffile.TiffFile(d + '/ink.tif'); p = t.pages[0]; th, tw = p.tilelength, p.tilewidth; nx = -(-p.shape[1] // tw)
A = np.zeros((n, n), np.uint8)
with open(d + '/ink.tif', 'rb') as f:
    for ty in range(y0 // th, (y0 + n - 1) // th + 1):
        for tx in range(x0 // tw, (x0 + n - 1) // tw + 1):
            i = ty * nx + tx; f.seek(p.dataoffsets[i]); a, _, _ = p.decode(f.read(p.databytecounts[i]), i); a = a.reshape(th, tw)
            ys, xs = ty * th, tx * tw; Y0, X0 = max(ys, y0), max(xs, x0); Y1, X1 = min(ys + th, y0 + n), min(xs + tw, x0 + n)
            A[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = a[Y0 - ys:Y1 - ys, X0 - xs:X1 - xs]
Image.fromarray(A).save(out); np.save(out[:-4] + '.npy', A)

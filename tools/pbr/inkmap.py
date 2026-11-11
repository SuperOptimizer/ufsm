# inkmap.py SEGDIR: mean published ink prediction per 1024^2 tile of ink.tif (and the masked-surface share) -> SEGDIR/inkmap.npy
import sys, numpy as np, tifffile
from concurrent.futures import ThreadPoolExecutor
d = sys.argv[1]; t = tifffile.TiffFile(d + '/ink.tif'); p = t.pages[0]
th, tw = p.tilelength, p.tilewidth; ny, nx = -(-p.shape[0] // th), -(-p.shape[1] // tw)
M = np.zeros((ny, nx, 2), np.float32)
fh = t.filehandle
def one(i):
    off, cnt = p.dataoffsets[i], p.databytecounts[i]
    if cnt == 0: return i, None
    with open(d + '/ink.tif', 'rb') as f: f.seek(off); raw = f.read(cnt)
    a, _, _ = p.decode(raw, i); a = a.reshape(th, tw)
    return i, (float((a > 127).mean()), float((a > 0).mean()))
with ThreadPoolExecutor(6) as ex:
    for i, r in ex.map(one, range(ny * nx)):
        if r: M[i // nx, i % nx] = r
np.save(d + '/inkmap.npy', M)
v = M[..., 0][M[..., 1] > 0.5]
print(d.split('/')[-1], 'tiles', ny, nx, 'surface tiles', int((M[..., 1] > 0.5).sum()), 'ink>0.5 share mean %.4f p90 %.4f max %.3f' % (v.mean(), np.percentile(v, 90), v.max()))

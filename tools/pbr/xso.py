# xso.py DIR DEPTH STEP: the row / column cross-sections of a sheet-pbr run with the recto (red) and verso (green) faces drawn
import sys, numpy as np, json
from PIL import Image
d, D, st = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); W = json.load(open(d + '/map.json'))['W']; K = int(2 * D / st) + 1
r = np.fromfile(d + '/recto_off.f32', np.float32).reshape(W, W); v = np.fromfile(d + '/verso_off.f32', np.float32).reshape(W, W)
for c in ['row', 'col']:
    a = np.fromfile(d + '/xs.%s.f32' % c, np.float32).reshape(W, K).T
    rr, vv = (r[W // 2, :], v[W // 2, :]) if c == 'row' else (r[:, W // 2], v[:, W // 2])
    im = np.stack([np.clip(a * 1.6, 0, 255)] * 3, -1).astype(np.uint8)
    for x in range(W):
        for val, col in ((rr[x], (255, 0, 0)), (vv[x], (0, 255, 0))):
            if np.isfinite(val):
                k = int(round((val + D) / st))
                if 0 <= k < K: im[k, x] = col
    Image.fromarray(im).save(d + '/xso_%s.png' % c)

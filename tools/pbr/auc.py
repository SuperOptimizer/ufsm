# auc.py OUTDIR INK.npy: AUC of each sheet-pbr map (and of each CT layer) for the published ink prediction (pos >= 160, neg <= 64),
# raw and detrended (minus a Gaussian, sigma argv[3] px, default 300 = 0.7 mm (letter strokes are ~0.5 mm)); AUC < 0.5 means the map is lower on ink
import sys, json, numpy as np
from scipy.ndimage import gaussian_filter
d, ink = sys.argv[1], np.load(sys.argv[2]).astype(np.float32); W = json.load(open(d + '/map.json'))['W']; ss = W // ink.shape[0]
pos, neg = ink >= 160, ink <= 64
rng = np.random.default_rng(0); SIG = float(sys.argv[3]) if len(sys.argv) > 3 else 300
def auc(a):
    a = a[::ss, ::ss] if ss > 1 else a
    m = np.isfinite(a); P = a[pos & m]; N = a[neg & m]
    if len(P) < 100 or len(N) < 100: return float('nan')
    P = rng.choice(P, min(len(P), 200000), replace=False); N = rng.choice(N, min(len(N), 200000), replace=False)
    from scipy.stats import rankdata; r = rankdata(np.concatenate([P, N]))   # average ranks: ties count half
    return (r[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N))
from scipy.ndimage import zoom
def detr(a):   # NaN-aware Gaussian trend on an 8x reduced grid (sigma >= 50 px), subtracted
    m = np.isfinite(a); b = np.where(m, a, 0).astype(np.float32); f = 8
    H = (a.shape[0] // f) * f; bs = b[:H, :H].reshape(H // f, f, H // f, f).mean((1, 3)); ms = m[:H, :H].reshape(H // f, f, H // f, f).mean((1, 3)).astype(np.float32)
    g = gaussian_filter(bs, SIG / f) / np.maximum(gaussian_filter(ms, SIG / f), 1e-6); g = zoom(g, f, order=1)
    out = np.full(a.shape, np.nan, np.float32); out[:H, :H] = a[:H, :H] - g; out[~m] = np.nan; return out
res = {}
for name in json.load(open(d + '/map.json'))['maps']:
    a = np.fromfile(d + '/%s.f32' % name, np.float32).reshape(W, W); res[name] = (auc(a), auc(detr(a)))
import glob
for f in glob.glob(d + '/layers_*.u8'):
    la, lb = map(int, f.split('layers_')[1][:-3].split('_')); L = np.fromfile(f, np.uint8).reshape(lb - la + 1, W, W).astype(np.float32)
    for l in range(L.shape[0]): res['layer%+d' % (la + l)] = (auc(L[l]), auc(detr(L[l])))
json.dump(res, open(d + '/auc.json', 'w'), indent=0)
print('pos %.3f neg %.3f of pixels' % (pos.mean(), neg.mean()))
for k, (a, b) in res.items():
    if not k.startswith('layer'): print('%-12s %.3f %.3f' % (k, a, b))
ly = sorted((int(k[5:]), v) for k, v in res.items() if k.startswith('layer'))
print('layers (t: raw / detrended): ' + ' '.join('%+d:%.2f/%.2f' % (t, a, b) for t, (a, b) in ly if t % 4 == 0))

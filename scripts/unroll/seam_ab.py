"""seam_ab.py W Y0C: per arm (A B C of seam_ab.sh) the mean |step| of phase and t0 across the 256- and 512-cell tile borders along y and x,
over the interior median (material cells only). Y0C: the box's y / x origin in field cells (level-2 voxels / 2)."""
import sys, numpy as np
W = sys.argv[1]; y0c = int(sys.argv[2]); sh = (256, 1024, 1024)
def load(a, p): return np.fromfile(f'{W}/{a}.{p}.raw', np.uint8).reshape(sh).astype(np.float32)
import os
for a in [x for x in 'ABC' if os.path.exists(f'{W}/{x}.pc.raw')]:   # (B, the 1152 window, needs a 24 GB card)
    L = {p: load(a, p) for p in ('pc', 'ps', 't0', 'w')}
    ph = np.angle((L['pc'] - 128) + 1j * (L['ps'] - 128)); mat = L['w'] > 0
    out = []
    for axis in (1, 2):
        for p in ('phase', 't0'):
            d = np.abs(np.angle(np.exp(1j * np.diff(ph, axis=axis)))) if p == 'phase' else np.abs(np.diff(L[p], axis=axis))
            mm = np.take(mat, range(1, sh[axis]), axis=axis) & np.take(mat, range(0, sh[axis] - 1), axis=axis)
            other = tuple(x for x in range(3) if x != axis)
            m = (d * mm).sum(axis=other) / np.maximum(mm.sum(axis=other), 1)
            b256 = [k for k in range(len(m)) if (y0c + k + 1) % 256 == 0]; b512 = [k for k in range(len(m)) if (y0c + k + 1) % 512 == 0]
            nb = [k for k in range(len(m)) if all((y0c + k + 1 + o) % 256 for o in (-1, 0, 1))]
            med = np.median(m[nb])
            out.append(f"{'yx'[axis - 1]} {p}: 256-borders x{np.mean(m[b256]) / med:.2f} 512-borders x{np.mean(m[b512]) / med:.2f} (interior median {med:.3f})")
    print(a, ' | '.join(out))

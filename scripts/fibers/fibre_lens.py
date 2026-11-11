"""fibre_lens.py LINES.fib UM: line-length statistics of a UFIB file (points in level-0 voxels of UM um): count, metres, median, mean,
arc-weighted median (half the traced arc lies in lines at least this long), p90, and the arc share in lines >= 1 mm and >= 2 mm.
The plain median of a traced set is pinned near fiber-trace's --min-len (48 prediction voxels = 0.46 mm at 9.6 um, 2026-11-11), so the
arc-weighted median and the >= 1 / 2 mm shares are the continuity measures."""
import struct, sys
import numpy as np
b = open(sys.argv[1], "rb").read(); um = float(sys.argv[2]); v, n, npt = struct.unpack("<3i", b[4:16]); o = 16
off = np.frombuffer(b, "<i4", n + 1, o); o += 4 * (n + 1) + n
pts = np.frombuffer(b, "<f4", 3 * npt, o).reshape(-1, 3).astype(float)
if n == 0: print("n 0"); sys.exit(0)
cs = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1) * um / 1000)]); L = cs[off[1:] - 1] - cs[off[:-1]]
s = np.sort(L); aw = s[np.searchsorted(np.cumsum(s), s.sum() / 2)]
print(f"n {n} total {L.sum() / 1000:.2f} m median {np.median(L):.2f} mean {L.mean():.2f} arc-median {aw:.2f} p90 {np.percentile(L, 90):.2f} "
      f">=1mm {L[L >= 1].sum() / L.sum():.3f} >=2mm {L[L >= 2].sum() / L.sum():.3f}")

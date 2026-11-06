# basespiral.py BOX Z0 Y0 X0 N AXIS AXS: current spiral (cal/PHerc1667 codes, dir +1) vs GT (dir -1) on GT voxels with our code within 1:
# area share on the box-mode sheet offset (same metric as psolve.py), and recall of our sheets at tol 1.
import sys, json, numpy as np
from scipy import ndimage
bd = sys.argv[1]; z0, y0, x0, n = map(int, sys.argv[2:6]); axp, axs = sys.argv[6], float(sys.argv[7])
gt = np.fromfile(bd + "/gt.raw", np.uint8).reshape(n, n, n); ou = np.fromfile(bd + "/ours.raw", np.uint8).reshape(n, n, n)
gs = (gt >= 1) & (gt <= 252); os_ = (ou >= 1) & (ou <= 252); oun = ndimage.maximum_filter(np.where(os_, ou, 0), size=3)
cp = json.load(open(axp))["control_points"]; cz = np.array([c["z"] for c in cp]) * axs; cy = np.array([c["y"] for c in cp]) * axs; cx = np.array([c["x"] for c in cp]) * axs
o = np.argsort(cz); cz, cy, cx = cz[o], cy[o], cx[o]
z, y, x = np.nonzero(gs & (oun > 0)); th = np.arctan2(y + y0 - np.interp(z + z0, cz, cy), x + x0 - np.interp(z + z0, cz, cx))
t0 = np.arctan2(np.sin(th).mean(), np.cos(th).mean()); th = t0 + np.angle(np.exp(1j * (th - t0)))
sg = np.rint((gt[z, y, x] - 1) / 18.0 + th / (2 * np.pi)).astype(int); so = np.rint((oun[z, y, x] - 1) / 18.0 - th / (2 * np.pi)).astype(int)
best = max(np.bincount(np.mod(s * so - sg, 14), minlength=14).max() for s in (1, -1)) / len(z)
print("%s: current spiral right sheet+turn %.4f on GT voxels it covers | covers %.3f of GT" % (bd, best, len(z) / gs.sum()))

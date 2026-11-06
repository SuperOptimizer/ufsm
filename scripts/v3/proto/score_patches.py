# score_patches.py DIR BOXDIR Z0 Y0 X0 NZ NY NX AXIS AXIS_SCALE DIR_GT
# VC3D tifxyz patches (box voxel coords) vs GT winding codes of the box (BOXDIR/gt.raw, uint8 winding_mod14, same grid):
#  recall: GT sheet voxels within 1 voxel of a densified patch; precision: densified patch voxels within 1 of GT / those within 8 of GT;
#  purity per patch: share of its GT-matched voxels on its dominant GT sheet (sheet = round(q_gt - dir theta / 2 pi) mod 14).
import sys, os, json, glob, numpy as np, tifffile
from scipy import ndimage
d, bd = sys.argv[1], sys.argv[2]; z0, y0, x0, nz, ny, nx = map(int, sys.argv[3:9]); axp, axs, sgn = sys.argv[9], float(sys.argv[10]), float(sys.argv[11])
gt = np.fromfile(os.path.join(bd, "gt.raw"), np.uint8).reshape(nz, ny, nx)
gs = (gt >= 1) & (gt <= 252)
cp = json.load(open(axp))["control_points"]; cz = np.array([c["z"] for c in cp]) * axs; cyv = np.array([c["y"] for c in cp]) * axs; cxv = np.array([c["x"] for c in cp]) * axs
o = np.argsort(cz); cz, cyv, cxv = cz[o], cyv[o], cxv[o]
def theta(z, y, x):
    gz = z + z0; ay = np.interp(gz, cz, cyv); ax = np.interp(gz, cz, cxv); return np.arctan2(y + y0 - ay, x + x0 - ax)
gsd1 = ndimage.binary_dilation(gs, iterations=1, structure=np.ones((3, 3, 3), bool))
gsd8 = ndimage.binary_dilation(gs, iterations=8)
# nearest GT code within 1 voxel: max-filter of codes (any GT code in the 3^3 nbhd; ties -> max)
gtn = ndimage.maximum_filter(np.where(gs, gt, 0), size=3)
cover = np.zeros_like(gs)
rows = []
for p in sorted(glob.glob(os.path.join(d, "*"))):
    if not os.path.exists(os.path.join(p, "x.tif")): continue
    X, Y, Z = (tifffile.imread(os.path.join(p, c + ".tif")).astype(np.float32) for c in "xyz")
    v = X > -1
    if v.sum() < 4: continue
    f = int(round(1 / json.load(open(os.path.join(p, "meta.json")))["scale"][0]))   # grid step in voxels
    Xi, Yi, Zi = (ndimage.zoom(A, f, order=1) for A in (X, Y, Z)); vi = ndimage.zoom(v.astype(np.float32), f, order=1) > 0.999
    VS = float(os.environ.get("VSCALE", "1")); ZO = float(os.environ.get("ZOFF", "0"))   # patch voxels -> box voxels (e.g. 2.4 um patches on the 9.6 GT grid)
    xi, yi, zi = (np.rint(A[vi] * VS + (ZO if c == "z" else 0)).astype(np.int64) for A, c in ((Xi, "x"), (Yi, "y"), (Zi, "z")))
    ins = (xi >= 0) & (yi >= 0) & (zi >= 0) & (xi < nx) & (yi < ny) & (zi < nz)
    xi, yi, zi = xi[ins], yi[ins], zi[ins]
    if not len(xi): continue
    cover[zi, yi, xi] = True
    near = gsd8[zi, yi, xi]; hit = gsd1[zi, yi, xi] & near
    c = gtn[zi, yi, xi][hit].astype(np.float64)
    if len(c):
        th = theta(zi[hit].astype(float), yi[hit].astype(float), xi[hit].astype(float))
        tr = np.arctan2(np.sin(th).mean(), np.cos(th).mean()); th = tr + np.angle(np.exp(1j * (th - tr)))   # unwrapped about the patch (no cut jump)
        q = (c - 1) / 18.0 - sgn * th / (2 * np.pi)
        s = np.mod(np.rint(q), 14).astype(int); h = np.bincount(s, minlength=14); pur = h.max() / h.sum()
    else: pur = np.nan
    rows.append((os.path.basename(p), int(ins.sum()), int(near.sum()), int(hit.sum()), pur))
cd = ndimage.binary_dilation(cover, iterations=1, structure=np.ones((3, 3, 3), bool))
rec = (gs & cd).sum() / max(1, gs.sum())
nn = sum(r[2] for r in rows); hh = sum(r[3] for r in rows)
pw = [r for r in rows if r[3] > 0]
wp = sum(r[4] * r[3] for r in pw) / max(1, sum(r[3] for r in pw))
print("patches %d | recall %.4f | precision %.4f (%d of %d voxels near GT) | purity (matched-weighted) %.4f | patches with purity < .95: %d of %d"
      % (len(rows), rec, hh / max(1, nn), hh, nn, wp, sum(1 for r in pw if r[4] < 0.95), len(pw)))

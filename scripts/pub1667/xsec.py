#!/usr/bin/env python3
"""CT xy cross-sections with our wraps and a published surface overlaid (QA images).

  xsec.py PUB_TIFXYZ OURS.f32 AXIS.json OUT_DIR --z 8000,16000,24000,32000 --ct ROOT KEY --ct-level 2 [--ufsm build/ufsm] [--crop 700] [--at 'y,x,size,scale;...']

PUB_TIFXYZ / OURS.f32 / AXIS.json as for surfcmp.py (level-0 voxels; OURS from ufsm wrap-dump --grid-um <level-0 um>). The published
surface is cut exactly (grid edges crossing the plane, linear interpolation); ours are the wrap vertices within half our vertex
spacing of the plane, labelled by wrap id. Per z: z<Z>_full.png (the whole slice, half size) and z<Z>_<deg>.png crops (1:1 CT voxels,
2x enlarged) at four angles about the axis. Published: blue, ours: orange (ids printed along a ray at the crop's angle).
"""
import os, subprocess, sys, json
import numpy as np
import tifffile
from PIL import Image, ImageDraw

def opt(k, d):
    a = sys.argv
    return a[a.index(k) + 1] if k in a else d

def read_pgm(p):
    with open(p, 'rb') as f:
        assert f.readline().strip() == b'P5'
        l = f.readline()
        while l.startswith(b'#'): l = f.readline()
        w, h = map(int, l.split()); mx = int(f.readline())
        return np.frombuffer(f.read(), np.uint8 if mx < 256 else '>u2').reshape(h, w)

def plane_cut(X, Y, Z, z0):
    pts = []
    for A0, A1 in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        z_a, z_b = Z[A0], Z[A1]; ok = (z_a > 0) & (z_b > 0) & ((z_a - z0) * (z_b - z0) <= 0) & (z_a != z_b)
        t = (z0 - z_a[ok]) / (z_b[ok] - z_a[ok])
        pts.append(np.stack([Y[A0][ok] + t * (Y[A1][ok] - Y[A0][ok]), X[A0][ok] + t * (X[A1][ok] - X[A0][ok])], 1))
    return np.concatenate(pts)

def main():
    pub, ours_f, axis_f, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    zs = [int(v) for v in opt('--z', '8000,16000,24000,32000').split(',')]
    a = sys.argv; i = a.index('--ct'); root, key = a[i + 1], a[i + 2]; lev = int(opt('--ct-level', '2')); f = 2 ** lev
    ufsm = opt('--ufsm', '/home/forrest/ufsm/build/ufsm'); crop = int(opt('--crop', '700')); half = float(opt('--half', '8'))
    pts = json.load(open(axis_f)); pts = pts['control_points'] if isinstance(pts, dict) else pts
    ax = np.array(sorted((p['z'], p['y'], p['x']) for p in pts))
    X = tifffile.imread(f'{pub}/x.tif'); Y = tifffile.imread(f'{pub}/y.tif'); Z = tifffile.imread(f'{pub}/z.tif')
    q = np.memmap(ours_f, dtype=np.float32, mode='r').reshape(-1, 5)
    zq = np.asarray(q[:, 0]); sel = {z0: np.flatnonzero(np.abs(zq - z0) < half) for z0 in zs}; del zq
    for z0 in zs:
        zl = int(round((z0 + 0.5) / f - 0.5)); pg = f'{out}/.ct{zl}.pgm'
        subprocess.run([ufsm, 'slice', root, key, str(zl), pg], check=True, stdout=subprocess.DEVNULL)
        ct = read_pgm(pg).astype(np.float32); os.remove(pg)
        lo, hi = np.percentile(ct[ct > 0], [1, 99.5]) if (ct > 0).any() else (0, 255)
        g = np.clip((ct - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        P = (plane_cut(X, Y, Z, z0) + 0.5) / f - 0.5           # level-L voxel coordinates
        O = np.asarray(q[sel[z0]]); Oy = (O[:, 1] + 0.5) / f - 0.5; Ox = (O[:, 2] + 0.5) / f - 0.5; Oid = O[:, 3].astype(int)
        cy = (np.interp(z0, ax[:, 0], ax[:, 1]) + 0.5) / f - 0.5; cx = (np.interp(z0, ax[:, 0], ax[:, 2]) + 0.5) / f - 0.5
        def draw(x0, y0, w, h, s, path, label_ang=None):
            im = Image.fromarray(g[y0:y0 + h, x0:x0 + w]).convert('RGB').resize((w * s, h * s), Image.NEAREST); dr = ImageDraw.Draw(im)
            r = max(1, s // 2)
            for yy, xx in zip(Oy, Ox):
                if x0 <= xx < x0 + w and y0 <= yy < y0 + h: X_, Y_ = (xx - x0 + 0.5) * s, (yy - y0 + 0.5) * s; dr.ellipse([X_ - r, Y_ - r, X_ + r, Y_ + r], fill=(0xeb, 0x68, 0x34))
            for yy, xx in P:
                if x0 <= xx < x0 + w and y0 <= yy < y0 + h: X_, Y_ = (xx - x0 + 0.5) * s, (yy - y0 + 0.5) * s; dr.ellipse([X_ - r, Y_ - r, X_ + r, Y_ + r], fill=(0x39, 0x87, 0xe5))
            if label_ang is not None:   # our wrap ids along the crop's central ray
                th = np.arctan2(Oy - cy, Ox - cx); m = (np.abs(np.angle(np.exp(1j * (th - label_ang)))) < 0.01) & (Ox >= x0) & (Ox < x0 + w) & (Oy >= y0) & (Oy < y0 + h)
                seen = set()
                for yy, xx, k in zip(Oy[m], Ox[m], Oid[m]):
                    if k in seen: continue
                    seen.add(k); dr.text(((xx - x0) * s + 6, (yy - y0) * s - 6), f'w{k}', fill=(255, 220, 120))
            dr.rectangle([0, 0, 520, 18], fill=(0, 0, 0))
            dr.text((4, 3), f'z {z0} (l0) / {zl} (l{lev}); blue: published unwrap, orange: ours (wraps_snap); 1 px = {f * 2.399 / s:.1f} um', fill=(255, 255, 255))
            im.save(path, optimize=True)
        H_, W_ = g.shape
        nz = np.argwhere(g > 0); y0, x0 = nz.min(0); y1, x1 = nz.max(0)
        draw(int(x0), int(y0), int(x1 - x0), int(y1 - y0), 1, f'{out}/z{z0}_full.png')
        # crops at four angles at mid radius of the published cut
        rr = np.hypot(P[:, 0] - cy, P[:, 1] - cx); rm = np.median(rr) if len(rr) else 300
        for deg in (0, 90, 180, 270):
            t = np.radians(deg); ccx, ccy = cx + rm * np.cos(t), cy + rm * np.sin(t)
            xa = int(np.clip(ccx - crop // 2, 0, W_ - crop)); ya = int(np.clip(ccy - crop // 2, 0, H_ - crop))
            draw(xa, ya, crop, crop, 2, f'{out}/z{z0}_a{deg:03d}.png', label_ang=t)
        for k, c in enumerate(opt('--at', '').split(';') if opt('--at', '') else []):   # extra crops: y,x (level-L voxels),size,scale
            yy, xx, sz, sc = (int(v) for v in c.split(','))
            ang = float(np.arctan2(yy - cy, xx - cx))
            draw(int(np.clip(xx - sz // 2, 0, W_ - sz)), int(np.clip(yy - sz // 2, 0, H_ - sz)), sz, sz, sc, f'{out}/z{z0}_at{yy}_{xx}.png', label_ang=ang)
        print(z0, 'pub cut points', len(P), 'our vertices', len(O), flush=True)

if __name__ == '__main__':
    sys.exit(main())

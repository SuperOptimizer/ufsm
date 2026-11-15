#!/usr/bin/env python3
"""reg791.py: residual of the published 2.399 um -> 7.91 um (2023, unmasked) PHerc1667 registration, measured on our CT.

Boxes of 64^3 level-2 voxels (9.596 um, 614 um a side) sit just inside the SAM2 mask (the CT band the fill must join).
Per box: our level-2 CT, and the 2023 slices sampled at the same points through the published affine (moving 2.399 l0 xyz ->
fixed 7.91 voxel xyz); 3-D phase correlation with a parabolic sub-voxel peak gives the leftover shift in um.
Also writes quantile pairs (ours vs 2023) over the boxes for the fill's intensity match.

usage: reg791.py TRANSFORM.json SLICEDIR CT_L4.raw OUT.tsv FC [FC ...]     (FC: fixed z slab centres; slices FC-48..FC+47)
env: UFSM (the ufsm binary), CTROOT, CTKEY
"""
import json, os, subprocess, sys, tempfile
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, map_coordinates

UFSM = os.environ.get('UFSM', '/home/forrest/ufsm/build/ufsm')
CTROOT = os.environ.get('CTROOT', '/vesuvius/ufsm/v2/ct')
CTKEY = os.environ.get('CTKEY', 'PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr')
L4 = (2318, 952, 952)
B = 64           # box edge, level-2 voxels
HALF = 48        # slab half thickness, fixed slices


def read_l2(z, y, x, n=B):
    with tempfile.NamedTemporaryFile(suffix='.raw') as f:
        subprocess.run([UFSM, 'read', CTROOT, CTKEY + '/2', str(z), str(y), str(x), str(n), str(n), str(n), f.name],
                       check=True, stdout=subprocess.DEVNULL)
        return np.fromfile(f.name, np.uint8).reshape(n, n, n)


def phasecorr(a, b):
    """shift s (zyx, voxels) with b(p) ~ a(p - s); peak height."""
    w = np.hanning(a.shape[0])
    w = w[:, None, None] * w[None, :, None] * w[None, None, :]
    fa = np.fft.fftn((a - a.mean()) / (a.std() + 1e-9) * w)
    fb = np.fft.fftn((b - b.mean()) / (b.std() + 1e-9) * w)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    c = np.fft.ifftn(r).real
    p = np.unravel_index(np.argmax(c), c.shape)
    s = []
    for ax in range(3):
        i = list(p); im = list(p); ip = list(p)
        im[ax] = (p[ax] - 1) % c.shape[ax]; ip[ax] = (p[ax] + 1) % c.shape[ax]
        y0, ym, yp = c[tuple(i)], c[tuple(im)], c[tuple(ip)]
        d = ym - 2 * y0 + yp
        off = 0.5 * (ym - yp) / d if d < 0 else 0.0
        v = p[ax] + off
        if v > c.shape[ax] / 2: v -= c.shape[ax]
        s.append(v)
    return np.array(s), float(c[p])


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def main():
    tr, sdir, l4p, outp = sys.argv[1:5]
    fcs = [int(v) for v in sys.argv[5:]]
    T = json.load(open(tr))
    M = np.array(T['transformation_matrix'], float)       # fixed_xyz = M[:, :3] @ moving_xyz + M[:, 3]
    l4 = np.memmap(l4p, np.uint8, 'r', shape=L4)
    rows, qa, qb = [], [], []
    for fc in fcs:
        sl = np.stack([tifffile.imread(f'{sdir}/{i:05d}.tif') for i in range(fc - HALF, fc + HALF)])
        print(f'slab {fc}: {sl.shape}', flush=True)
        cands = []
        for ang in np.arange(0, 360, 15):
            # moving l0 z giving fixed z = fc at the scroll centre, then refine with the box's own x, y
            cz = (fc - M[2, 3] - M[2, 0] * 7600 - M[2, 1] * 7600) / M[2, 2]
            z4 = int(round(cz / 16))
            for _ in range(3):
                pl = l4[z4] > 0
                if pl.sum() < 1000: break
                d = distance_transform_edt(pl)
                yy, xx = np.nonzero(pl)
                cy, cx = yy.mean(), xx.mean()
                dy, dx = np.sin(np.radians(ang)), np.cos(np.radians(ang))
                hit = None
                for r in np.arange(480, 0, -1.0):          # walk inward; first point >= 14 l4 voxels inside
                    y, x = int(round(cy + r * dy)), int(round(cx + r * dx))
                    if 0 <= y < L4[1] and 0 <= x < L4[2] and d[y, x] >= 14:
                        hit = (y, x); break
                if hit is None: break
                y0 = hit[0] * 16 + 8; x0 = hit[1] * 16 + 8
                zm = (fc - M[2, 3] - M[2, 0] * x0 - M[2, 1] * y0) / M[2, 2]
                nz4 = int(round(zm / 16))
                if nz4 == z4: break
                z4 = nz4
            if hit is not None and pl.sum() >= 1000:
                cands.append((ang, zm, hit[0] * 16 + 8, hit[1] * 16 + 8, float(d[hit])))
        for ang, zm, ym, xm, dedge in cands:
            z2, y2, x2 = int(zm / 4) - B // 2, int(ym / 4) - B // 2, int(xm / 4) - B // 2
            if min(z2, y2, x2) < 0 or z2 + B > 9269 or max(y2, x2) + B > 3808: continue
            ours = read_l2(z2, y2, x2).astype(np.float32)
            g = np.mgrid[0:B, 0:B, 0:B].astype(np.float64)
            pz, py, px = (g[0] + z2) * 4 + 1.5, (g[1] + y2) * 4 + 1.5, (g[2] + x2) * 4 + 1.5
            fx = M[0, 0] * px + M[0, 1] * py + M[0, 2] * pz + M[0, 3]
            fy = M[1, 0] * px + M[1, 1] * py + M[1, 2] * pz + M[1, 3]
            fz = M[2, 0] * px + M[2, 1] * py + M[2, 2] * pz + M[2, 3] - (fc - HALF)
            if fz.min() < 0 or fz.max() > 2 * HALF - 1:
                print(f'  ang {ang}: fixed z {fz.min():.1f}..{fz.max():.1f} outside the slab', flush=True); continue
            fix = map_coordinates(sl, [fz, fy, fx], order=1, mode='nearest').astype(np.float32)
            inside = ours > 0
            fin = inside.mean()
            s, pk = phasecorr(ours, fix)
            n0 = ncc(ours[inside], fix[inside])
            si = np.round(s).astype(int)
            n1 = ncc(ours, np.roll(fix, tuple(-si), (0, 1, 2)))
            um = np.linalg.norm(s) * 9.596
            rows.append((fc, ang, int(zm), ym, xm, dedge * 38.38, fin, s[0], s[1], s[2], um, pk, n0, n1))
            print(f'  fc {fc} ang {ang:5.0f} l0 zyx {int(zm)} {ym} {xm} edge {dedge*38.38:5.0f} um  in {fin:.2f}  shift zyx '
                  f'{s[0]:+.2f} {s[1]:+.2f} {s[2]:+.2f} l2vox = {um:5.1f} um  peak {pk:.3f}  ncc {n0:.3f} -> {n1:.3f}', flush=True)
            if fin > 0.95 and pk > 0.05:
                qs = np.linspace(0.005, 0.995, 199)
                qa.append(np.quantile(ours[inside], qs)); qb.append(np.quantile(fix[inside], qs))
        del sl
    with open(outp, 'w') as f:
        f.write('fc\tang\tz0\ty0\tx0\tedge_um\tinside\tdz\tdy\tdx\tshift_um\tpeak\tncc0\tncc_shift\n')
        for r in rows: f.write('\t'.join(f'{v:.4g}' if isinstance(v, float) else str(v) for v in r) + '\n')
    if qa:
        np.savez(outp.replace('.tsv', '_q.npz'), ours=np.array(qa), fixed=np.array(qb))
    r = np.array([x[10] for x in rows]); pk = np.array([x[11] for x in rows])
    good = pk > 0.05
    print(f'boxes {len(rows)}, confident (peak > 0.05) {good.sum()}: shift um median {np.median(r[good]):.1f} '
          f'p90 {np.percentile(r[good], 90):.1f} max {r[good].max():.1f}; mean dz dy dx (l2 vox) '
          f'{np.mean([x[7] for x, g in zip(rows, good) if g]):+.2f} {np.mean([x[8] for x, g in zip(rows, good) if g]):+.2f} '
          f'{np.mean([x[9] for x, g in zip(rows, good) if g]):+.2f}')


if __name__ == '__main__':
    main()

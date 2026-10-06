#!/usr/bin/env python3
"""Band field (whole-sheet labels) of a box from the winding_mod14 raster: reference implementation and previews.

Every voxel within R of a labelled recto surface gets b = q(nearest surface voxel) + 0.5 * side, side = +1 if the voxel
is farther from the umbilicus than that surface voxel (or level with it), else -1. Between consecutive wraps q, q+1 both
bounding rectos give b = q + 0.5, so b is constant inside a band and jumps by one turn exactly at each recto.
Unknown (not supervised): farther than R from any surface, CT == 0, near a conflict voxel (two surfaces > 1/4 turn apart
in one voxel), and the span around a "missing wrap" (b jumps by >= 0.5 away from any surface: the two nearest rectos are
2+ turns apart). Works on the 4.8 um label grid. Band instances = connected components of voxels joined across
non-jumps.

usage: band_field.py --box z,y,x,nz,ny,nx --out DIR [--codes STORE] [--preview 1]
Writes DIR/band.npz (k = band in code steps of 1/18 turn mod 252, -1 = unknown, at 4.8 um, origin = box origin // 2) and previews.
"""
import argparse, json, subprocess, tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

UFSM = Path(__file__).resolve().parent.parent / 'build/ufsm'
CODES = '/vesuvius/ufsm/gt/paris4-band-20261005/q-4p8.zarr'   # winding_mod14 raster (ufsm raster --value q.tif)
CT_ROOT, CT = '/vesuvius/usrm/volcomp', 'PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr'
AXIS = '/vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json'
REFERENCE = '/vesuvius/ufsm/gt/paris4-sheet-20261003/geometry/reference.json'
R_NATIVE = 80   # flood radius (native voxels): about 3/4 of the local turn pitch (90-128 native voxels per turn)


def read(store, key, lo, size, path):
    r = subprocess.run([str(UFSM), 'read', str(store), key, *map(str, lo), *map(str, size), str(path), '--threads', '8'], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    return np.fromfile(path, np.uint8).reshape(size)


STEPS = 18          # code steps per turn (winding_mod14: 1 + (round(q * 18) mod 252))
PERIOD = 252        # steps per code period (14 turns)
HALF = STEPS // 2   # half a turn


def mdiff(a, b):
    """modular difference of band / winding steps, in (-126, 126]"""
    return (np.asarray(a, np.int32) - np.asarray(b, np.int32) + PERIOD // 2) % PERIOD - PERIOD // 2


def band_field(codes, origin_l, axis, pitch_of_z, R=R_NATIVE):
    """codes: uint8 winding_mod14 [nz,ny,nx] on the label grid, origin origin_l (label voxels).
    Returns band steps k (int16, -1 = unknown): b = q(nearest surface) +- half a turn, in code steps mod 252."""
    seed = (codes > 0) & (codes < 255)
    conflict = codes == 255
    dist, ind = ndi.distance_transform_edt(~seed, return_indices=True)
    dist *= 2.0   # native voxels
    sz, sy, sx = ind
    ks = codes[sz, sy, sx].astype(np.int32) - 1
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in codes.shape], indexing='ij')
    def radius(z, y, x):
        zn = 2.0 * (z + origin_l[0]) + 0.5
        cy = np.interp(zn, axis[:, 0], axis[:, 1]); cx = np.interp(zn, axis[:, 0], axis[:, 2])
        return np.hypot(2.0 * (y + origin_l[1]) + 0.5 - cy, 2.0 * (x + origin_l[2]) + 0.5 - cx)
    side = np.where(radius(zz, yy, xx) >= radius(sz, sy, sx), HALF, -HALF)
    k = ((ks + side) % PERIOD).astype(np.int16)
    del zz, yy, xx, sz, sy, sx, ind, ks, side
    unknown = dist > R
    unknown |= ndi.binary_dilation(conflict, iterations=1)
    # missing wraps: the band value jumps by >= half a turn between neighbours that are both off the surfaces
    jump = np.zeros(codes.shape, bool)
    far = dist > 2.0
    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[a] = slice(0, -1); s1[a] = slice(1, None)
        j = (np.abs(mdiff(k[tuple(s0)], k[tuple(s1)])) >= HALF) & far[tuple(s0)] & far[tuple(s1)]
        jump[tuple(s0)] |= j; jump[tuple(s1)] |= j
    if jump.any():
        zc = 2.0 * (np.arange(codes.shape[0]) + origin_l[0]) + 0.5
        span = int(np.ceil(0.75 * float(np.median(pitch_of_z(zc))) / 2.0))   # label voxels
        unknown |= ndi.distance_transform_edt(~jump) <= span
    k[unknown] = -1
    return k, dict(seed_fraction=float(seed.mean()), conflict_voxels=int(conflict.sum()), jump_voxels=int(jump.sum()),
                   unknown_fraction=float(unknown.mean()))


def instances(k):
    """connected components of known voxels joined across neighbours whose band differs by < half a turn"""
    known = k >= 0
    cut = ~known
    for a in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[a] = slice(0, -1); s1[a] = slice(1, None)
        d = (np.abs(mdiff(k[tuple(s0)], k[tuple(s1)])) >= HALF) & known[tuple(s0)] & known[tuple(s1)]
        cut[tuple(s1)] |= d   # the outer voxel of each jump becomes the cut
    lab, n = ndi.label(~cut)
    return lab, n


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--box', required=True); a.add_argument('--out', required=True); a.add_argument('--codes', default=CODES)
    a.add_argument('--preview', type=int, default=1); a.add_argument('--R', type=float, default=R_NATIVE)
    g = a.parse_args()
    box = [int(v) for v in g.box.split(',')]; o, n = np.array(box[:3]), np.array(box[3:])
    if (o % 2).any() or (n % 2).any(): raise SystemExit('box must be even')
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    halo = int(np.ceil(g.R / 2)) + 4
    lo = o // 2 - halo; size = n // 2 + 2 * halo
    axis = json.loads(Path(AXIS).read_text())['control_points']
    axis = np.array(sorted([[p['z'], p['y'], p['x']] for p in axis]));
    knots = np.array(json.loads(Path(REFERENCE).read_text())['knots'], float)   # pitch (native voxels per turn) by z
    pitch_of_z = lambda z: np.interp(z, knots[:, 0], knots[:, 3])
    with tempfile.TemporaryDirectory() as d:
        codes = read(g.codes, '4.8', lo, size, Path(d) / 'c.raw')
        ct = read(CT_ROOT, CT + '/1', o // 2, n // 2, Path(d) / 'ct.raw')
    seeds = codes[(codes > 0) & (codes < 255)]
    if not seeds.size: raise SystemExit('no labelled surfaces in the box')
    k, info = band_field(codes, lo, axis, pitch_of_z, g.R)
    k = k[halo:-halo, halo:-halo, halo:-halo]
    k[ct == 0] = -1
    lab, nlab = instances(k)
    info.update(box=box, R_native=g.R, instances=int(nlab), known_fraction=float((k >= 0).mean()))
    np.savez_compressed(out / 'band.npz', k=k, origin_4p8=o // 2, instances=lab.astype(np.int32))
    (out / 'band.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    if g.preview:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        for name, sl in (('z', (k.shape[0] // 2, slice(None), slice(None))), ('y', (slice(None), k.shape[1] // 2, slice(None)))):
            fig, ax = plt.subplots(1, 3, figsize=(21, 7))
            ax[0].imshow(ct[sl], cmap='gray'); ax[0].set_title('CT (4.8 um) %s-slice' % name)
            par = np.where(k[sl] >= 0, (k[sl] // STEPS) % 2, np.nan).astype(float)
            ax[1].imshow(ct[sl], cmap='gray'); ax[1].imshow(par, cmap='coolwarm', alpha=0.45, vmin=0, vmax=1); ax[1].set_title('band parity (unknown = no colour)')
            rng = np.random.default_rng(1); lut = rng.random((nlab + 1, 3)); lut[0] = 0
            ax[2].imshow(lut[lab[sl]]); ax[2].set_title('band instances (%d in box)' % nlab)
            for x in ax: x.axis('off')
            fig.tight_layout(); fig.savefig(out / ('preview_%s.png' % name), dpi=80); plt.close(fig)


if __name__ == '__main__':
    main()

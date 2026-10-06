#!/usr/bin/env python3
"""Per-vertex winding coordinate q (turns) for the selected Paris4 meshes, for the band (whole-sheet) labels.

The aligned winding geometry (tools/build_sheet_geometry.py -> geometry.alignment.npz) holds q only at decimated
training points, and none inside the withheld boxes (plus guard). Here every mesh is lifted again on the same decimated
grid with the same unwrap (tools/sheet_geometry.py:unwrap_mesh, withheld region included), each connected patch gets the
integer turn offset that makes it agree with the aligned points (exact coordinate matches), and every full-resolution
vertex gets q = q(nearest decimated node) + wrapped phase difference. Patches without enough reliable aligned matches,
or whose offset is not a clean integer, are dropped (q = NaN).

Writes <out>/<segment>/ with symlinks to the source x/y/z.tif and q.tif (float32), plus <out>/audit.json.
usage: build_band_meshes.py --out DIR [--geometry-dir /vesuvius/ufsm/gt/paris4-sheet-20261003]
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import tifffile

from sheet_geometry import unwrap_mesh, atomic_json, digest
from build_sheet_geometry import load_axis

MESHES = '/vesuvius/ufsm/gt/paris4-all-surfaces/aws'
SEGMENTS = Path(__file__).resolve().parent.parent / 'configs/paris4-segments-20260623.txt'
AXIS = '/vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json'
VOLUME = '20260411134726'


def phase_of(xyz, axis):
    cy = np.interp(xyz[..., 0], axis[:, 0], axis[:, 1]); cx = np.interp(xyz[..., 0], axis[:, 0], axis[:, 2])
    return np.arctan2(xyz[..., 1] - cy, xyz[..., 2] - cx) / (2 * np.pi)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', required=True); a.add_argument('--geometry-dir', default='/vesuvius/ufsm/gt/paris4-sheet-20261003')
    a.add_argument('--edge-limit', type=float, default=512); a.add_argument('--min-matches', type=int, default=20)
    g = a.parse_args()
    out = Path(g.out); out.mkdir(parents=True, exist_ok=True)
    gd = Path(g.geometry_dir)
    galign = np.load(gd / 'geometry.alignment.npz'); gaudit = json.loads((gd / 'geometry.audit.json').read_text())
    if gaudit.get('winding_direction', 1) != 1: raise SystemExit('expected winding_direction 1')
    if gaudit['axis_sha256'] != digest(AXIS) or gaudit['segments_sha256'] != digest(SEGMENTS): raise SystemExit('geometry inputs changed')
    axis = load_axis(AXIS)
    unanchored = set(gaudit.get('unanchored_components', []))
    segments = [s.strip() for s in SEGMENTS.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    audit = dict(version=1, geometry=str(gd), alignment_sha256=digest(gd / 'geometry.alignment.npz'), segments=[])
    comp_base = 0
    for seg, meta in zip(segments, gaudit['meshes']):
        assert meta['segment'] == seg
        stride, ncomp = meta['stride'], meta['components']
        root = next(Path(MESHES).glob(f'PHercParis4/segments/{seg}/mesh/*-on-{VOLUME}-2.4um.tifxyz'))
        raw = np.stack([tifffile.imread(root / f'{c}.tif') for c in 'zyx'], axis=-1)
        h, w = raw.shape[:2]
        valid = np.isfinite(raw).all(-1) & (raw > 0).all(-1)
        coarse = raw[::stride, ::stride].astype(np.float64); cvalid = valid[::stride, ::stride]
        qc, compc, _, bad = unwrap_mesh(coarse, cvalid, axis, edge_limit=g.edge_limit)
        # aligned points of this mesh: global components [comp_base, comp_base + ncomp)
        sel = (galign['component'] >= comp_base) & (galign['component'] < comp_base + ncomp)
        key = lambda p: (p[..., 0] * 1e6 + p[..., 1]) * 1e6 + p[..., 2]   # exact-match key (coordinates are identical floats)
        ak = key(galign['xyz'][sel]); order = np.argsort(ak); ak = ak[order]
        aq = galign['q'][sel][order]; arel = galign['reliable'][sel][order]; acomp = galign['component'][sel][order]
        ck = key(coarse[cvalid & np.isfinite(qc)]); pos = np.searchsorted(ak, ck); pos = np.minimum(pos, len(ak) - 1)
        hit = len(ak) > 0
        matched = (ak[pos] == ck) if hit else np.zeros(len(ck), bool)
        qn = qc[cvalid & np.isfinite(qc)]; cn = compc[cvalid & np.isfinite(qc)]
        offsets, comps = {}, []
        for c in np.unique(cn):
            m = (cn == c) & matched
            m &= arel[pos] & ~np.isin(acomp[pos], list(unanchored)) if hit else m
            info = dict(component=int(c), nodes=int((cn == c).sum()), matches=int(m.sum()))
            if m.sum() >= g.min_matches:
                d = aq[pos[m]] - qn[m]; off = float(np.round(np.median(d)))
                frac = float(np.mean(np.abs(d - off) < 0.05))
                info.update(offset=off, integer_fraction=frac, residual=float(np.median(np.abs(d - off))))
                if frac >= 0.95: offsets[int(c)] = off
            info['kept'] = int(c) in offsets
            comps.append(info)
        qa = np.full(qc.shape, np.nan)
        for c, off in offsets.items(): qa[compc == c] = qc[compc == c] + off
        # full resolution: nearest decimated node + wrapped phase difference
        rr, cc = np.nonzero(valid)
        nr = np.minimum(np.rint(rr / stride).astype(np.int64), qa.shape[0] - 1); nc = np.minimum(np.rint(cc / stride).astype(np.int64), qa.shape[1] - 1)
        node_q = qa[nr, nc]
        pf = phase_of(raw[rr, cc].astype(np.float64), axis); pn = phase_of(coarse[nr, nc], axis)
        q = np.full((h, w), np.nan, np.float32)
        q[rr, cc] = (node_q + ((pf - pn + .5) % 1 - .5)).astype(np.float32)
        # continuity check between adjacent full-resolution vertices
        dq = np.abs(np.diff(q, axis=1)); jumps = float(np.nanmean(dq > 0.25)) if np.isfinite(dq).any() else 0.0
        lo_w, hi_w = (int(x) for x in seg.split('-w')[1].split('-')) if '-w' in seg else (None, None)
        fq = q[np.isfinite(q)]; inname = float(np.mean((np.floor(fq) + 27 >= lo_w) & (np.floor(fq) + 27 <= hi_w))) if fq.size else 0.0
        d = out / seg; d.mkdir(exist_ok=True)
        for c in 'xyz':
            (d / f'{c}.tif').unlink(missing_ok=True); os.symlink(root / f'{c}.tif', d / f'{c}.tif')
        (d / 'meta.json').unlink(missing_ok=True); os.symlink(root / 'meta.json', d / 'meta.json')
        tifffile.imwrite(d / 'q.tif', q)
        rec = dict(segment=seg, source=str(root), stride=stride, vertices=int(valid.sum()), kept=int(np.isfinite(q).sum()),
                   kept_fraction=float(np.isfinite(q).sum() / max(1, valid.sum())), inconsistent_components=[int(b) for b in bad],
                   components=comps, q_range=[float(fq.min()), float(fq.max())] if fq.size else None,
                   name_wrap_match=inname, adjacent_jump_fraction=jumps, q_sha256=digest(d / 'q.tif'))
        audit['segments'].append(rec)
        print('%s: kept %.4f of %d vertices, q %s, name match %.4f, jumps %.2e, components kept %d/%d' %
              (seg, rec['kept_fraction'], rec['vertices'], rec['q_range'], inname, jumps, len(offsets), len(comps)), flush=True)
        comp_base += ncomp
    tot = sum(s['vertices'] for s in audit['segments']); kept = sum(s['kept'] for s in audit['segments'])
    audit['kept_fraction'] = kept / tot
    atomic_json(out / 'audit.json', audit)
    print('total kept %.4f of %d vertices' % (kept / tot, tot))


if __name__ == '__main__':
    main()

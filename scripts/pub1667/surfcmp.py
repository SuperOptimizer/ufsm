#!/usr/bin/env python3
"""Our unroll's wraps vs a published continuous unwrap surface (tifxyz) of the same scan, CPU only.

  surfcmp.py PUB_TIFXYZ OURS.f32 AXIS.json OUT_DIR [--um 2.399] [--strip 0,39] [--rev-every 2] [--workers 16]

PUB_TIFXYZ: x.tif / y.tif / z.tif in level-0 voxels of our scan (invalid: z <= 0 or -1).
OURS.f32: `ufsm wrap-dump WRAPS OURS.f32 --z -10,1e9 --every 1 --grid-um <level-0 um>` records (z, y, x, id, column / width).
AXIS.json: umbilicus control points ("x", "y", "z") in level-0 voxels.

Distances: per published vertex the nearest of our vertices; signed s = (q - p) . n_p with n_p the published normal oriented away from
the axis (s > 0: our surface lies outside the published one); the plane distance |s| is used when the tangential part is below
--tan-um (the nearest-vertex error of our ~40 um vertex grid), else the point distance. The reverse the same with the roles swapped
(published normals in both cases). Sheet identity: the published surface's continuous turn T_pub (axis angle unwrapped along its
columns) minus u, the angle fraction from our wraps' theta cut (column 0), is an integer per published sheet along a ray from the axis;
our wrap id minus that integer, per matched vertex, is one scroll-wide constant where both follow the same sheets ("same sheet"); +-1
where one of the two surfaces sits on the neighbouring sheet (a sheet switch of ours or of theirs; geometry alone cannot say which).
Outputs OUT_DIR/summary.json, pub_turns.tsv, our_wraps.tsv, pub_dist.npz (per published grid vertex: d, s, our id, T_pub, sheet offset).
"""
import json, math, os, sys, time
import numpy as np
import tifffile
from scipy.spatial import cKDTree

def opt(k, d):
    a = sys.argv
    return a[a.index(k) + 1] if k in a else d

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def axis_load(path):
    pts = json.load(open(path))
    pts = pts['control_points'] if isinstance(pts, dict) else pts
    a = np.array(sorted((p['z'], p['y'], p['x']) for p in pts), dtype=np.float64)
    return a

def axis_at(ax, z):
    return np.interp(z, ax[:, 0], ax[:, 1]), np.interp(z, ax[:, 0], ax[:, 2])

def main():
    if len(sys.argv) < 5:
        print(__doc__); return 2
    pub_dir, ours_f, axis_f, out = sys.argv[1:5]
    um = float(opt('--um', '2.399')); s0, s1 = (int(v) for v in opt('--strip', '0,39').split(','))
    rev_every = int(opt('--rev-every', '2')); W = int(opt('--workers', '16')); tan_um = float(opt('--tan-um', '60'))
    os.makedirs(out, exist_ok=True)
    ax = axis_load(axis_f)

    # ---- published surface
    X = tifffile.imread(f'{pub_dir}/x.tif').astype(np.float32); Y = tifffile.imread(f'{pub_dir}/y.tif').astype(np.float32)
    Z = tifffile.imread(f'{pub_dir}/z.tif').astype(np.float32)
    H, Wd = Z.shape
    valid = (Z > 0) & np.isfinite(Z) & (X > 0) & (Y > 0)
    log(f'pub grid {H}x{Wd}, valid {valid.mean():.4f}')
    P = np.stack([Z, Y, X], -1)
    def cdiff(A, axis):
        d = np.zeros_like(A); v = np.zeros(A.shape[:2], bool)
        if axis == 1:
            d[:, 1:-1] = (A[:, 2:] - A[:, :-2]) * 0.5; v[:, 1:-1] = valid[:, 2:] & valid[:, :-2]
        else:
            d[1:-1] = (A[2:] - A[:-2]) * 0.5; v[1:-1] = valid[2:] & valid[:-2]
        return d, v
    du, vu = cdiff(P, 1); dv, vv = cdiff(P, 0)
    n = np.cross(du, dv); nl = np.linalg.norm(n, axis=-1)
    area = nl.copy()                                        # vox^2 per vertex (central differences / 2 each side)
    ok = valid & vu & vv & (nl > 0)
    n = n / np.maximum(nl, 1e-9)[..., None]
    cy, cx = axis_at(ax, Z.astype(np.float64))
    ry, rx = Y - cy, X - cx; rad = np.hypot(ry, rx)
    out_dot = n[..., 1] * ry + n[..., 2] * rx
    n[out_dot < 0] *= -1
    theta = np.arctan2(ry, rx)
    # per-column circular mean angle, unwrapped along the columns -> continuous turn
    c = np.where(valid, np.cos(theta), 0).sum(0); s = np.where(valid, np.sin(theta), 0).sum(0); cnt = valid.sum(0)
    colok = cnt >= 8
    phi = np.full(Wd, np.nan); phi[colok] = np.unwrap(np.arctan2(s[colok], c[colok]))
    phi = np.interp(np.arange(Wd), np.flatnonzero(colok), phi[colok])
    Phi = phi[None, :] + np.angle(np.exp(1j * (theta - phi[None, :])))
    # orientation: turn grows outward
    rr = rad[valid]; pp = Phi[valid]; sgn = 1.0 if np.corrcoef(pp[::97], rr[::97])[0, 1] > 0 else -1.0
    Tp = sgn * Phi / (2 * np.pi); Tp -= np.nanmin(np.where(valid, Tp, np.nan))
    dirang = sgn                                           # +1: outward = increasing atan2 angle
    log(f'pub turns {np.nanmax(np.where(valid, Tp, np.nan)):.2f}, radius {rr.min()*um/1000:.2f}-{rr.max()*um/1000:.2f} mm, angle sign {dirang:+.0f}')
    # lengths / area
    step = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1); sv = valid[:, 1:] & valid[:, :-1]
    rows = np.flatnonzero(valid.sum(1) > 0.5 * Wd)
    col_len = np.where(sv, step, np.nan)
    length_mm = np.nanmedian(np.nansum(col_len[rows], 1)) * um / 1000
    sum_dir = np.nanmean(np.where(sv, step, np.nan)) * um
    area_cm2 = float(area[ok].sum(dtype=np.float64)) * um * um * 1e-8
    log(f'pub length (median full row) {length_mm/10:.1f} cm, mean column step {sum_dir:.1f} um, area {area_cm2:.1f} cm2')

    vi = np.flatnonzero(ok.ravel())
    Pv = P.reshape(-1, 3)[vi].astype(np.float64); Nv = n.reshape(-1, 3)[vi]; Av = area.ravel()[vi]; Tv = Tp.ravel()[vi]; Rv = rad.ravel()[vi]
    Thv = theta.ravel()[vi]; okm = ok.copy()
    del P, du, dv, vu, vv, n, nl, area, cy, cx, ry, rx, out_dot, theta, Phi, Tp, rad, step, sv, col_len, X, Y, Z

    # ---- ours
    rec = np.fromfile(ours_f, dtype=np.float32).reshape(-1, 5)
    Q = rec[:, :3].astype(np.float64); qid = rec[:, 3].astype(np.int32); qcol = rec[:, 4].copy(); del rec
    log(f'ours {len(Q)} vertices, wraps {qid.min()}..{qid.max()}')
    qcy, qcx = axis_at(ax, Q[:, 0]); qth = np.arctan2(Q[:, 1] - qcy, Q[:, 2] - qcx); qrad = np.hypot(Q[:, 1] - qcy, Q[:, 2] - qcx)
    ids = np.unique(qid)
    # our wrap order vs radius
    sub = slice(None, None, 20); qid_s, qrad_s, qcol_s, qth_s = qid[sub], qrad[sub], qcol[sub], qth[sub]
    medr = {int(k): float(np.median(qrad_s[qid_s == k])) for k in ids}
    our_out = 1 if np.corrcoef(list(medr.keys()), list(medr.values()))[0, 1] > 0 else -1
    cut = {}
    for k in ids:
        m = (qid_s == k) & (qcol_s < 0.003)
        cut[int(k)] = float(np.angle(np.exp(1j * qth_s[m]).mean())) if m.any() else 0.0
    lut = np.zeros(ids.max() - ids.min() + 1); [lut.__setitem__(int(k) - ids.min(), cut[int(k)]) for k in ids]
    fr = np.mod(dirang * (qth - lut[qid - ids.min()]), 2 * np.pi) / (2 * np.pi)
    Tq = our_out * qid + fr
    log(f'our ids increase {"outward" if our_out > 0 else "inward"}')

    # ---- published -> ours
    t0 = time.time(); tq = cKDTree(Q, leafsize=32, balanced_tree=False, compact_nodes=False); log(f'tree ours {time.time()-t0:.0f} s')
    dpp, j = tq.query(Pv, k=1, workers=W)
    sgnd = np.einsum('ij,ij->i', Q[j] - Pv, Nv)
    tan = np.sqrt(np.maximum(dpp * dpp - sgnd * sgnd, 0))
    d = np.where(tan * um < tan_um, np.abs(sgnd), dpp) * um; sgnd *= um
    jid = qid[j]
    # gauge: the published sheet index along a ray from the axis, with our cut angle as the ray origin: n_pub = T - u (u = angle fraction
    # from our cut, in the outward angle direction) is an integer + one constant; our wrap id there should be n_pub + one scroll-wide offset
    cut0 = float(np.angle(np.mean(np.exp(1j * np.array(list(cut.values()))))))
    u = np.mod(dirang * (Thv - cut0), 2 * np.pi) / (2 * np.pi)
    D = Tv - u; c0 = float(np.angle(np.exp(2j * np.pi * D).mean()) / (2 * np.pi))
    npub = np.round(D - c0).astype(np.int32)
    off = jid - npub
    close = d < 100
    nearcut = (u < 0.01) | (u > 0.99)                      # our cut seam: the nearest vertex may sit across the cut (id +-1, same sheet)
    log(f'pub->ours done, c0 {c0:.3f}')

    # ---- ours -> published
    t0 = time.time(); tp = cKDTree(Pv, leafsize=32, balanced_tree=False, compact_nodes=False); log(f'tree pub {time.time()-t0:.0f} s')
    sel = np.arange(0, len(Q), rev_every)
    rd, rj = tp.query(Q[sel], k=1, workers=W)
    rs = np.einsum('ij,ij->i', Q[sel] - Pv[rj], Nv[rj]); rt = np.sqrt(np.maximum(rd * rd - rs * rs, 0))
    rdd = np.where(rt * um < tan_um, np.abs(rs), rd) * um; rs *= um
    # ours outside the published surface's footprint (its z / angle extent) are not "misses": flag those whose nearest pub vertex is a border vertex
    border = np.zeros((H, Wd), bool); vv2 = okm
    border[:-1] |= vv2[:-1] & ~vv2[1:]; border[1:] |= vv2[1:] & ~vv2[:-1]; border[:, :-1] |= vv2[:, :-1] & ~vv2[:, 1:]; border[:, 1:] |= vv2[:, 1:] & ~vv2[:, :-1]
    border[0] = border[-1] = True; border[:, 0] = border[:, -1] = True
    rb = border.ravel()[vi][rj]
    del tp
    log('ours->pub done')

    def q(v, p): return float(np.percentile(v, p)) if len(v) else float('nan')
    summ = {'pub': pub_dir, 'pub_grid': [H, Wd], 'pub_valid_vertices': int(len(vi)), 'pub_area_cm2': area_cm2, 'pub_length_cm': float(length_mm / 10),
            'pub_turns': float(Tv.max()), 'pub_radius_mm': [float(Rv.min() * um / 1000), float(Rv.max() * um / 1000)], 'c0': c0}
    A = Av
    for th in (50, 100, 200):
        summ[f'pub_cov_{th}'] = float(A[d < th].sum() / A.sum())
    summ.update(pub_d_med=q(d, 50), pub_d_p90=q(d, 90), pub_s_med=q(sgnd[close], 50), pub_s_p10=q(sgnd[close], 10), pub_s_p90=q(sgnd[close], 90))
    gm = close & ~nearcut
    vals, cnts = np.unique(off[gm], return_counts=True); g0 = int(vals[np.argmax(cnts)])
    summ['gauge_offset'] = g0; summ['gauge_share'] = {int(v): float(c_ / cnts.sum()) for v, c_ in zip(vals, cnts) if c_ / cnts.sum() > 0.001}
    instrip = (qid[sel] >= s0) & (qid[sel] <= s1); inside = ~rb
    for nm, m in (('all', inside), ('strip', inside & instrip)):
        for th in (50, 100, 200):
            summ[f'ours_{nm}_cov_{th}'] = float((rdd[m] < th).mean())
        summ[f'ours_{nm}_d_med'] = q(rdd[m], 50); summ[f'ours_{nm}_d_p90'] = q(rdd[m], 90); summ[f'ours_{nm}_vertices'] = int(m.sum())
        summ[f'ours_{nm}_s_med'] = q(rs[m & (rdd < 100)], 50)
    summ['ours_outside_pub_footprint'] = float(rb.mean())
    json.dump({k_: (float(v_) if isinstance(v_, (np.floating, np.integer)) else v_) for k_, v_ in summ.items()}, open(f'{out}/summary.json', 'w'), indent=1)
    for k_, v_ in summ.items(): print(f'  {k_}: {v_}')

    # per published turn
    kt = np.floor(Tv).astype(np.int32)
    with open(f'{out}/pub_turns.tsv', 'w') as f:
        f.write('pub_turn\tarea_cm2\tr_med_mm\tcov50\tcov100\tcov200\td_med_um\td_p90_um\ts_med_um\tour_id\tid_share\tsecond_id\tsecond_share\tsame_sheet\tid_plus1\tid_minus1\n')
        for k in range(kt.min(), kt.max() + 1):
            m = kt == k
            if m.sum() < 100: continue
            a = A[m]; mc = m & close
            ids_, c_ = np.unique(jid[mc], return_counts=True); o = np.argsort(-c_)
            top = (int(ids_[o[0]]), c_[o[0]] / c_.sum()) if len(o) else (-999, 0)
            sec = (int(ids_[o[1]]), c_[o[1]] / c_.sum()) if len(o) > 1 else (-999, 0)
            mg = mc & ~nearcut; ng = max(mg.sum(), 1)
            f.write(f'{k}\t{a.sum()*um*um*1e-8:.2f}\t{np.median(Rv[m])*um/1000:.2f}\t{a[d[m]<50].sum()/a.sum():.3f}\t{a[d[m]<100].sum()/a.sum():.3f}\t'
                    f'{a[d[m]<200].sum()/a.sum():.3f}\t{np.median(d[m]):.0f}\t{np.percentile(d[m],90):.0f}\t{np.median(sgnd[mc]) if mc.any() else float("nan"):.0f}\t'
                    f'{top[0]}\t{top[1]:.3f}\t{sec[0]}\t{sec[1]:.3f}\t{(off[mg]==g0).sum()/ng:.3f}\t{(off[mg]==g0+1).sum()/ng:.3f}\t{(off[mg]==g0-1).sum()/ng:.3f}\n')
    # per our wrap
    with open(f'{out}/our_wraps.tsv', 'w') as f:
        f.write('our_id\tvertices\tr_med_mm\tinside_pub_footprint\tcov50\tcov100\tcov200\td_med_um\ts_med_um\tpub_turn\tturn_share\n')
        rq = qid[sel]; rTp = np.floor(Tv[rj]).astype(np.int32)
        for k in ids:
            m = (rq == k); mi = m & inside
            if mi.sum() < 10:
                f.write(f'{k}\t{m.sum()}\t{medr[int(k)]*um/1000:.2f}\t{mi.sum()/max(m.sum(),1):.3f}\t\t\t\t\t\t\t\n'); continue
            mc = mi & (rdd < 100); t_, c_ = np.unique(rTp[mc], return_counts=True)
            tt = (int(t_[np.argmax(c_)]), c_.max() / c_.sum()) if len(c_) else (-999, 0)
            f.write(f'{k}\t{m.sum()}\t{medr[int(k)]*um/1000:.2f}\t{mi.sum()/m.sum():.3f}\t{(rdd[mi]<50).mean():.3f}\t{(rdd[mi]<100).mean():.3f}\t{(rdd[mi]<200).mean():.3f}\t'
                    f'{np.median(rdd[mi]):.0f}\t{np.median(rs[mc]) if mc.any() else float("nan"):.0f}\t{tt[0]}\t{tt[1]:.3f}\n')
    np.savez_compressed(f'{out}/pub_dist.npz', shape=np.array([H, Wd]), vi=vi.astype(np.int64), d=d.astype(np.float32), s=sgnd.astype(np.float32),
                        id=jid.astype(np.int16), T=Tv.astype(np.float32), off=np.clip(off - g0, -100, 100).astype(np.int8), npub=npub.astype(np.int16), nearcut=nearcut, area=Av.astype(np.float32))
    log('written', out)

if __name__ == '__main__':
    sys.exit(main())

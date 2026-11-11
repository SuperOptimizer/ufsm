"""Shared helpers of the fibre analysis scripts (analysis only, not part of the pipeline)."""
import json, sqlite3
import numpy as np

Q = ("select b.fiber_id, b.first_segment, b.points, f.family, f.length from block_bounds r join blocks b on b.id = r.id join fibers f on f.id = b.fiber_id "
     "where r.max_x >= ? and r.min_x <= ? and r.max_y >= ? and r.min_y <= ? and r.max_z >= ? and r.min_z <= ? order by b.fiber_id, b.first_segment")

class Map:
    """AFV native xyz <-> our level-L zyx. tr: published transform.json (ours L0 xyz -> AFV xyz), or None when the AFV is native to our scan."""
    def __init__(self, level, tr=None):
        self.sc = 2.0 ** level; self.M = None
        if tr:
            self.M = np.array(json.load(open(tr))["transformation_matrix"], dtype=np.float64)[:3]; self.Mi = np.linalg.inv(self.M[:, :3])
    def to_level(self, xyz):
        if self.M is not None: xyz = (xyz - self.M[:, 3]) @ self.Mi.T
        return ((xyz + 0.5) / self.sc - 0.5)[:, ::-1]
    def box_afv(self, box):
        bz, by, bx, nz, ny, nx = box
        c = np.array([[bz + i * nz, by + j * ny, bx + k * nx] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)
        xyz = (c[:, ::-1] + 0.5) * self.sc - 0.5
        if self.M is not None: xyz = xyz @ self.M[:, :3].T + self.M[:, 3]
        return xyz.min(0), xyz.max(0)

def open_afv(path):
    db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    vs = json.loads(dict(db.execute("select key, value from metadata where key = 'root'"))["root"])["voxel_size_mm"]
    return db, vs

def runs_in_box(db, mp, box, step=1.0, min_len=2.0):
    """Fibre runs meeting the box, resampled every `step` level voxels: list of (fiber_id, fam 0 H / 1 V / 2 ?, fibre length (native voxels),
    points (n, 3) level zyx global, unit tangents)."""
    lo, hi = mp.box_afv(box); raw = []; pf = ps = None
    for fid, seg, blob, fam, flen in db.execute(Q, (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])):
        p = np.frombuffer(blob, dtype="<f8").reshape(-1, 3)
        if raw and fid == pf and seg == ps + 256: raw[-1][3] = np.vstack([raw[-1][3], p[1:]])
        else: raw.append([fid, {"H": 0, "V": 1}.get(fam, 2), flen, p])
        pf, ps = fid, seg
    out = []
    for fid, fam, flen, p in raw:
        v = mp.to_level(p); d = np.linalg.norm(np.diff(v, axis=0), axis=1); keep = np.concatenate([[True], d > 1e-9]); v = v[keep]
        if len(v) < 2: continue
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(v, axis=0), axis=1))])
        if s[-1] < min_len: continue
        t = np.arange(0, s[-1], step); r = np.stack([np.interp(t, s, v[:, k]) for k in range(3)], 1)
        tg = np.gradient(r, axis=0); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        out.append((fid, fam, flen, r, tg))
    return out

wrap = lambda v: v - np.floor(v + 0.5)

def phase_stats(runs, ph, box, D=(16, 32, 64, 128, 256), hyst=0.75, S=1, sampler=None):
    """Fibre runs against a band-phase raster box (uint8, 0..251 = turn x 252, 255 unknown). Per family f (0 H, 1 V, 2 ?):
    arc[f] labelled arc (level voxels), sw[f] sheet switches with hysteresis (after a 9-sample median; a there-and-back excursion counts 2),
    net[f] persistent switches (|round(end - start)| of each labelled stretch, medians of 8 samples), pair[f][di] = n, |dphi| < 0.25, < 0.5."""
    from scipy.ndimage import median_filter
    bz, by, bx, nz, ny, nx = box; nd = len(D)
    out = dict(arc=np.zeros(3), sw=np.zeros(3), net=np.zeros(3), pair=np.zeros((3, nd, 3)), pts=0, lab=0, nfib=np.zeros(3, int), fsw=np.zeros(3, int))
    for fid, fam, flen, r, tg in runs:
        q = np.rint(r - [bz, by, bx]).astype(np.int64)
        inb = np.all((q >= 0) & (q < np.array([nz, ny, nx])), axis=1)
        if S > 1: q = q // S   # ph holds one value per S^3 voxels (fit-phase-box --step S)
        v = np.full(len(r), 255, np.uint8)
        if sampler is not None:   # sampler(points (n, 3) global level zyx) -> uint8 phase (255 unknown), e.g. wrap_phase below
            if inb.any(): v[inb] = sampler(r[inb])
        else: v[inb] = ph[q[inb, 0], q[inb, 1], q[inb, 2]]
        ok = v <= 251; out["pts"] += int(inb.sum()); out["lab"] += int(ok.sum())
        if ok.sum() < 16: continue
        phi = v / 252.0; edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))); fs = 0; used = False
        for s0, s1 in zip(edges[::2], edges[1::2]):
            n = s1 - s0
            if n < 16: continue
            used = True
            um = median_filter(np.concatenate([[0], np.cumsum(wrap(np.diff(phi[s0:s1])))]), size=9, mode="nearest")
            c = 0
            if np.ptp(um) > hyst:
                b = um[0]
                for x in um:
                    if abs(x - b) > hyst: k = round(x - b); c += abs(k); b += k
            out["arc"][fam] += n; out["sw"][fam] += c; fs += c
            out["net"][fam] += abs(round(float(np.median(um[-8:]) - np.median(um[:8]))))
            for di, d in enumerate(D):
                if n <= d: continue
                st = max(1, d // 4); dd = np.abs(um[d::st] - um[:-d:st])
                out["pair"][fam, di] += [len(dd), np.sum(dd < 0.25), np.sum(dd < 0.5)]
        if used: out["nfib"][fam] += 1; out["fsw"][fam] += fs > 0
    return out

def offset_clusters(gt_codes, our_codes, our_phase, R=4, nmax=200000, seed=1):
    """GT-based score of OUR winding labels in one box (a proxy of the bench's ids / pair accuracy): at GT code voxels x, our winding
    W(x) = q(nearest our code within R) + phi - (phi >= 0.5) (our phase phi within 0.3 of a surface), offset o = W - q_gt (mod 14 turns).
    One spiral gauge <=> one integer offset over the box. Returns n, the share of the largest integer cluster (ids-like), sum p_k^2
    (P two labelled points agree, pair-like), the circular concentration of the fractional offset, the sign used."""
    sub = gt_codes[::2, ::2, ::2]; idx = np.argwhere((sub >= 1) & (sub <= 252)) * 2   # every 2nd voxel per axis: small temporaries (page faults are slow on the loaded box)
    if len(idx) < 500: return None
    rng = np.random.default_rng(seed)
    if len(idx) > nmax: idx = idx[rng.choice(len(idx), nmax, replace=False)]
    sh = np.array(gt_codes.shape); m = np.all((idx >= R) & (idx < sh - R), axis=1); idx = idx[m]
    phi = our_phase[tuple(idx.T)]; m = (phi <= 251) & ((phi < 76) | (phi > 176)); idx = idx[m]; phi = phi[m] / 252.0
    if len(idx) < 500: return None
    offs = [(dz, dy, dx) for dz in range(-R, R + 1) for dy in range(-R, R + 1) for dx in range(-R, R + 1) if dz * dz + dy * dy + dx * dx <= R * R]
    offs.sort(key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
    oc = np.zeros(len(idx), np.int32)
    for o in offs:
        todo = oc == 0
        if not todo.any(): break
        c = our_codes[tuple((idx[todo] + o).T)]; c = np.where((c >= 1) & (c <= 252), c, 0); oc[todo] = c
    m = oc > 0; idx = idx[m]; phi = phi[m]; oc = oc[m]
    if len(idx) < 500: return None
    qg = (gt_codes[tuple(idx.T)].astype(float) - 1) / 18; Wo = (oc.astype(float) - 1) / 18 + phi - (phi >= 0.5)
    best = None
    for sgn in (1, -1):
        o = np.mod(Wo - sgn * qg, 14); z = np.exp(2j * np.pi * o).mean(); fr = np.angle(z) / (2 * np.pi)
        k = np.rint(o - fr).astype(int) % 14; p = np.bincount(k, minlength=14) / len(k)
        res = (len(k), p.max(), float(np.sum(p * p)), abs(z), sgn)
        if best is None or res[2] > best[2]: best = res
    return best

def wrap_phase(trees, pts, R=40.0):
    """Band phase (pair rule) of points between the two nearest wrap sheets, from wrap vertices (ufsm wrap-dump). trees = ((tree, q) of the
    even-id wraps, (tree, q) of the odd-id wraps): cKDTrees of the vertices (zyx) and their along-spiral coordinate q = -id + column / width
    (continuous along the spiral: the end of wrap k + 1 joins the start of wrap k; radially adjacent sheets differ by about 1 and, away
    from the theta cut, in id parity). The nearer of the two parities' nearest vertices is sheet 1 (d1, q1), the other sheet 2 when
    0.5 <= |q2 - q1| < 1.5: phase = frac(sign(q2 - q1) d1 / (d1 + d2)); 255 otherwise (no second sheet within R, or the cut)."""
    (ta, qa), (tb, qb) = trees
    da, ia = ta.query(pts, k=1, distance_upper_bound=R); db_, ib = tb.query(pts, k=1, distance_upper_bound=R)
    ok = np.isfinite(da) & np.isfinite(db_); ia = np.where(ok, ia, 0); ib = np.where(ok, ib, 0)
    a1 = da <= db_; d1 = np.where(a1, da, db_); d2 = np.where(a1, db_, da); q1 = np.where(a1, qa[ia], qb[ib]); q2 = np.where(a1, qb[ib], qa[ia])
    dq = q2 - q1; ok &= (np.abs(dq) >= 0.5) & (np.abs(dq) < 1.5)
    out = np.full(len(pts), 255, np.uint8)
    with np.errstate(invalid="ignore", divide="ignore"): ph = np.sign(dq) * d1 / (d1 + d2)
    ph = ph - np.floor(ph); out[ok] = np.minimum(251, np.floor(ph[ok] * 252)).astype(np.uint8)
    return out

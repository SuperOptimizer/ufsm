"""fibgt.py AFV PHASE.raw --box z,y,x,nz,ny,nx --level L [--tr T.json] [--tag NAME] [--out rows.tsv]
Does a fibre stay in one sheet of a winding-phase raster? (analysis only; python3 -I.) PHASE.raw: uint8 box (`ufsm read`) of a band
phase raster (0..251 = turn x 252 outward from the labelled recto face, 255 unknown) - the human GT, or `surf-labels` of OUR wraps
(then the same numbers judge our spiral). Along every fibre run, sampled each level voxel, the phase is unwrapped along the path
(labelled stretches only; a stretch ends at an unknown voxel); a fibre inside one sheet keeps the unwrapped phase within a fraction
of a turn, a sheet switch moves it by +-1.
Prints per family and fibre-length class: labelled arc (m), switches per metre (hysteresis 0.75 turn after a 9-sample median),
pair tables at arc distance d (share |d phase| < 0.25 / < 0.5 turn), the tangent's angle out of the sheet plane (grad phase), and
the phase (depth in the sheet) histogram per family."""
import argparse, sys
sys.path.insert(0, "/home/forrest/ufsm-fib/tools")
import numpy as np
from scipy.ndimage import median_filter
from fiblib import Map, open_afv, runs_in_box

ap = argparse.ArgumentParser()
ap.add_argument("afv"); ap.add_argument("phase"); ap.add_argument("--box", required=True); ap.add_argument("--level", type=int, default=2)
ap.add_argument("--tr"); ap.add_argument("--tag", default="box"); ap.add_argument("--out"); ap.add_argument("--um", type=float, default=9.6)
ap.add_argument("--d", default="8,16,32,64,128,256,512"); ap.add_argument("--hyst", type=float, default=0.75)
a = ap.parse_args()
box = tuple(map(int, a.box.split(","))); bz, by, bx, nz, ny, nx = box
ph = np.fromfile(a.phase, dtype=np.uint8).reshape(nz, ny, nx)
mp = Map(a.level, a.tr); db, vs = open_afv(a.afv)
runs = runs_in_box(db, mp, box)
D = [int(x) for x in a.d.split(",")]
LC = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]   # fibre length classes, mm
def lclass(mm): return next(i for i, (lo, hi) in enumerate(LC) if lo <= mm < hi)
wrap = lambda v: v - np.floor(v + 0.5)
# accumulators [fam 0..2][len class]
arc = np.zeros((3, 5)); sw = np.zeros((3, 5)); nfib = np.zeros((3, 5), int); nsw_f = np.zeros((3, 5), int)
pair = np.zeros((3, len(D), 4))   # n, |d| < 0.25, |d| < 0.5, sum |d|
hist = np.zeros((3, 20)); ang = [[], [], []]; tot_pts = 0; tot_lab = 0
spc = [[], [], []]
for fid, fam, flen, r, tg in runs:
    q = np.rint(r - [bz, by, bx]).astype(np.int64)
    inb = np.all((q >= 2) & (q < np.array([nz, ny, nx]) - 2), axis=1)
    v = np.full(len(r), 255, np.uint8); v[inb] = ph[q[inb, 0], q[inb, 1], q[inb, 2]]
    ok = v <= 251; tot_pts += int(inb.sum()); tot_lab += int(ok.sum())
    if ok.sum() < 8: continue
    lc = lclass(flen * vs); phi = v / 252.0
    hist[fam] += np.histogram(phi[ok], bins=20, range=(0, 1))[0]
    # grad phase at a subsample (central difference over +-2 voxels, wrapped)
    idx = np.nonzero(ok)[0][::8]; g = np.zeros((len(idx), 3)); good = np.ones(len(idx), bool)
    for dim in range(3):
        o = np.zeros(3, np.int64); o[dim] = 2
        va = ph[tuple((q[idx] - o).T)]; vb = ph[tuple((q[idx] + o).T)]
        good &= (va <= 251) & (vb <= 251); g[:, dim] = wrap((vb.astype(float) - va) / 252.0) / 4
    gn = np.linalg.norm(g, axis=1); good &= gn > 1e-4
    if good.any():
        ang[fam].append(np.degrees(np.arcsin(np.clip(np.abs(np.sum(g[good] * tg[idx][good], axis=1)) / gn[good], 0, 1))))
        spc[fam].append(1 / gn[good])
    # labelled stretches
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
    fsw = 0
    for s0, s1 in zip(edges[::2], edges[1::2]):
        n = s1 - s0
        if n < 8: continue
        u = np.concatenate([[0], np.cumsum(wrap(np.diff(phi[s0:s1])))])
        um = median_filter(u, size=9, mode="nearest")
        b = um[0]; c = 0
        dv = um - b
        if np.abs(dv).max() > a.hyst:
            for x in um:
                if abs(x - b) > a.hyst: k = round(x - b); c += abs(k); b += k
        arc[fam, lc] += n; sw[fam, lc] += c; fsw += c
        for di, d in enumerate(D):
            if n <= d: continue
            dd = np.abs(um[d::max(1, d // 4)] - um[:-d:max(1, d // 4)])
            pair[fam, di] += [len(dd), np.sum(dd < 0.25), np.sum(dd < 0.5), dd.sum()]
    nfib[fam, lc] += 1; nsw_f[fam, lc] += fsw > 0
m_per = a.um * 1e-6
print(f"{a.tag}: {len(runs)} runs, {tot_pts} points in the box, {tot_lab} with a phase ({tot_lab / max(1, tot_pts):.3f})")
rows = []
for f, fn in enumerate("HV?"):
    if arc[f].sum() == 0: continue
    for l in list(range(5)) + [-1]:
        A = arc[f].sum() if l < 0 else arc[f, l]; S = sw[f].sum() if l < 0 else sw[f, l]; N = nfib[f].sum() if l < 0 else nfib[f, l]; NS = nsw_f[f].sum() if l < 0 else nsw_f[f, l]
        if A == 0: continue
        nm = "all" if l < 0 else (f"{LC[l][0]}-{LC[l][1] if LC[l][1] < 1e8 else 'inf'}mm")
        print(f"  {fn} len {nm:>8}: labelled arc {A * m_per:8.3f} m, switches {int(S):6d} = {S / (A * m_per):8.1f} /m ; fibres {N:6d}, with a switch {NS:6d} ({NS / max(1, N):.3f})")
        rows.append((a.tag, fn, nm, A * m_per, int(S), N, NS))
    for di, d in enumerate(D):
        n, a25, a50, sm = pair[f, di]
        if n: print(f"    pairs d {d:4d} vox ({d * a.um / 1000:.2f} mm): n {int(n):8d}  |dphi|<0.25 {a25 / n:.4f}  <0.5 {a50 / n:.4f}  mean |dphi| {sm / n:.3f}")
    if ang[f]:
        A2 = np.concatenate(ang[f]); S2 = np.concatenate(spc[f])
        print(f"    tangent out of the sheet plane (deg): median {np.median(A2):.1f}, p90 {np.percentile(A2, 90):.1f}, share > 30 deg {np.mean(A2 > 30):.3f}  (n {len(A2)}; sheet spacing median {np.median(S2):.1f} vox)")
    h = hist[f] / hist[f].sum(); c = np.sum(h * np.exp(2j * np.pi * (np.arange(20) + 0.5) / 20))
    print(f"    phase hist: circ-mean {np.angle(c) / (2 * np.pi) % 1:.3f} R {abs(c):.3f} | " + " ".join(f"{x:.3f}" for x in h))
if a.out:
    with open(a.out, "a") as o:
        for r in rows: o.write("\t".join(str(x) for x in r) + "\n")

#!/usr/bin/env python3
"""scripts/univ/unseen_boxes.py OUT.tsv [SCROLL ...]: the unseen-scroll boxes of the universal net's label-free eval (2026-11-12).

Scrolls with no labels and no fits of ours anywhere (not in any training source, no unroll / corpus fit, no ink / fibre / GT labels):
default the 9 corpus scans below, CT local under /vesuvius/ufsm/v2/ct. Boxes of N^3 level-0 voxels (N 512: 4.4 / 4.8 mm at
8.64 / 9.36 um, the wbench box scale), placed CPU-side and deterministically from the scan's level-4 (16x) volume and its corpus axis:
  - papyrus = level-4 CT >= the scan's Otsu threshold; the z range = where a slice holds >= 20 % of the peak papyrus count;
  - K (3) boxes per scroll, one per z level, z levels ZF tried in order (25-68 % of that range: off the ends; ~24 samples to pair), candidates at
    16 angles x radius 0.35 / 0.5 / 0.65 of that slice's papyrus radius (p90 from the axis), never overlapping a box already picked;
  - kept: the box inside the volume, its papyrus share in [0.2, 0.75]; picked: the share closest to 0.45 (dense sheets, not solid / void).
Columns: name scroll root key um level z y x nz ny nx axis papyrus_share r_frac
"""
import json, os, subprocess, sys
import numpy as np

U = os.environ.get('U', '/home/forrest/ufsm/build/ufsm'); ROOT = '/vesuvius/ufsm/v2/ct'; COR = '/vesuvius/ufsm/corpus'
LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../corpus/list.tsv')
DEF = ['PHerc0268', 'PHerc0800', 'PHerc0826', 'PHerc0175B', 'PHerc0306B', 'PHerc0483A', 'PHerc0490A', 'PHerc0846A', 'PHerc0343']
N, LV = int(os.environ.get('N', 512)), 4
ZF = [float(x) for x in os.environ.get('ZF', '0.45,0.3,0.6,0.38,0.52,0.25,0.68').split(',')]   # z levels tried in order (fractions of the papyrus z range)
K = int(os.environ.get('K', 3))   # boxes per scroll

def otsu(v):
    h = np.bincount(v.ravel(), minlength=256).astype(np.float64); p = h / h.sum(); w = np.cumsum(p); m = np.cumsum(p * np.arange(256))
    return int(np.argmax((m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)))

def main():
    out = sys.argv[1]; scrolls = sys.argv[2:] or DEF; rows = []
    lst = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open(LIST) if not l.startswith('scroll')}
    for sc in scrolls:
        key, um = lst[sc][2], float(lst[sc][3]); f = 1 << LV
        info = subprocess.run([U, 'info', ROOT, key], capture_output=True, text=True).stdout
        sh = [l.split('shape')[1].split('shard')[0].split('x') for l in info.splitlines() if l.strip().startswith(f'{LV} ')][0]; sh = [int(s) for s in sh]
        raw = f'/tmp/ub_{sc}.raw'
        if not os.path.exists(raw): subprocess.run([U, 'read', ROOT, f'{key}/{LV}', '0', '0', '0', *map(str, sh), raw, '--threads', '8'], check=True, capture_output=True)
        v = np.fromfile(raw, np.uint8).reshape(sh); thr = otsu(v[v > 0]); pap = v >= thr
        cnt = pap.sum((1, 2)); zz = np.nonzero(cnt >= 0.2 * cnt.max())[0]; z0, z1 = zz[0], zz[-1]; n = N / f
        ax = json.load(open(f'{COR}/{sc}/axis.json'))['control_points']; az = np.array([p['z'] for p in ax]) / f
        picked = []
        for zf in ZF:
            if len(picked) >= K: break   # one box per z level, spread over the scroll's length; boxes never overlap
            zc = int(z0 + zf * (z1 - z0)); best = None
            ay = np.interp(zc, az, np.array([p['y'] for p in ax]) / f); axx = np.interp(zc, az, np.array([p['x'] for p in ax]) / f)
            yy, xx = np.nonzero(pap[zc]); R = np.percentile(np.hypot(yy - ay, xx - axx), 90)
            for rf in (0.35, 0.5, 0.65):
                for k in range(16):
                    t = 2 * np.pi * k / 16; cy, cx = ay + rf * R * np.sin(t), axx + rf * R * np.cos(t)
                    b = [int(round(zc - n / 2)), int(round(cy - n / 2)), int(round(cx - n / 2))]
                    if min(b) < 0 or b[0] + n > sh[0] or b[1] + n > sh[1] or b[2] + n > sh[2]: continue
                    if any(all(abs(b[i] - q[i]) < n for i in range(3)) for q in picked): continue
                    s_ = float(pap[b[0]:b[0] + int(n), b[1]:b[1] + int(n), b[2]:b[2] + int(n)].mean())
                    if 0.2 <= s_ <= 0.75 and (best is None or abs(s_ - 0.45) < abs(best[0] - 0.45)): best = (s_, rf, b)
            if best is None: continue
            s_, rf, b = best; picked.append(b); b0 = [c * f for c in b]
            rows.append([f'U{sc[5:]}{"abcdef"[len(picked) - 1]}', sc, ROOT, key, f'{um:g}', '0', *map(str, b0), str(N), str(N), str(N), f'{COR}/{sc}/axis.json', f'{s_:.3f}', f'{rf:.2f}'])
            print('\t'.join(rows[-1]), file=sys.stderr)
    with open(out, 'w') as fo:
        fo.write('# name\tscroll\troot\tkey\tum\tlevel\tz\ty\tx\tnz\tny\tnx\taxis\tpapyrus_share\tr_frac   (scripts/univ/unseen_boxes.py)\n')
        for r in rows: fo.write('\t'.join(r) + '\n')

main()

#!/usr/bin/env python3
"""holdouts-20261114 selection (datafix-20261114): the existing boxes (wbench, ink eval crops, fibre HOLD, the 0139 / 0500P2 cross-res
maps) plus, per scan where a source stays under ~8 % of its labels, one full-yx z slab chosen from holdout-cover's z histograms so every
source on that scan lands nearest 8-12 %. Over-covered sets trimmed (0841 ink 49 %: its boxes greedily to ~10 %; 1299's bottom third
replaced by a slab). 0139: the slab is chosen jointly over the 9.362 and 2.399 um scans through the xres affine (z nearly axis aligned:
cz = 0.257 fz + 623, x / y tilt < 60 coarse rows -> a 300-row fine margin); the 2.399 um ink eval crops mapped back to 9.362 um."""
import json, glob, subprocess, sys, collections
H = '/vesuvius/ufsm/datafix-20261114/ho'; U = H + '/ufsm.hc'
base = json.load(open(H + '/base.json'))['boxes']
ctb = lambda k: k.rstrip('/').split('/')[-1]
# per source: name -> (ct, lev, cells, held, ZB, hist)
src = {}
for f in [H + '/zh-base.txt'] + sorted(glob.glob(H + '/one/*.zh')):
    for ln in open(f):
        t = ln.split();
        if len(t) < 7: continue
        src[t[0]] = dict(ct=t[1], lev=int(t[2]), cells=float(t[3]), held=float(t[4]), zb=int(t[5]), h=[float(v) for v in t[6:]])
# per-box shares (alone), from the one/*.txt outputs and cover-base.txt
def per_box(txt, name):
    out = []; on = False
    for ln in open(txt):
        if not ln.startswith(' '): on = ln.split()[0] == name; continue
        if on and ln.strip().startswith('box'): t = ln.replace('[', ' ').replace(']', ' ').split(); out.append(([int(v) for v in t[2:8]], float(t[-1])))
    return out
W = json.load(open('/vesuvius/ufsm/univ/scratch1/sources-scratch1-ph80.json'))
full = {s['name']: s for s in W['sources']}
def shape(name):
    s = full[name]; r = subprocess.run([U, 'info', s['root'], s['ct'] + '/0'], capture_output=True, text=True).stdout + subprocess.run([U, 'info', s['root'], s['ct']], capture_output=True, text=True).stdout
    import re; m = re.search(r'shape[^0-9]*(\d+)[, x]+(\d+)[, x]+(\d+)', r); return [int(m.group(i)) for i in (1, 2, 3)] if m else None
def pen(c): return 0 if 0.08 <= c <= 0.12 else (0.08 - c) * 3 if c < 0.08 else c - 0.12
def useg(h, a, b):   # the share in rows [a, b) of a 256-row histogram
    s = 0.0
    for q in range(max(0, int(a // 256)), min(len(h), int(-(-b // 256)))):
        lo, hi = max(a, q * 256), min(b, (q + 1) * 256); s += h[q] * max(0, hi - lo) / 256
    return s
def best_slab(groups, tmin, nz):   # groups: list of (source, rowmap fn(a, b) -> rows on its scan)
    best = None
    for t in range(tmin, max(tmin, nz // 3) + 1, 256):
        for a in range(0, max(1, nz - t + 1), 256):
            L = sum(pen(src[n]['held'] + useg(src[n]['h'], *fn(a, a + t))) for n, fn in groups)
            if best is None or L < best[0] - 1e-9: best = (L, a, t)
    return best
boxes = [b for b in base]
# 0841: greedy subset of its 6 boxes to ~10 % (alone shares; the boxes do not overlap)
pb = per_box(H + '/one/PHerc0841-u2ink.txt', 'PHerc0841-u2ink'); keep = []; c = 0
for l0, sh in sorted(pb, key=lambda v: v[1]):
    if c >= 0.08: break
    keep.append(l0); c += sh
boxes = [b for b in boxes if ctb(b['ct']) != src['PHerc0841-u2ink']['ct'] or b['l0'] in keep]
src['PHerc0841-u2ink']['held'] = c; print('0841: kept %d of %d boxes, %.3f' % (len(keep), len(pb), c))
# 0139 / 0500P2 fine-scan ink crops mapped onto the 9.362 um scans (the other scan of the same papyrus)
for reg, cfine in (('PHerc0139-20260102150214', '20260102150214-2.399um-0.2m-78keV-masked.zarr'), ('PHerc0500P2', '20250526151718-2.215um-0.4m-111keV-masked.zarr')):
    rj = json.load(open('/vesuvius/ufsm/reg/xres/%s/reg.json' % reg))
    for b in list(boxes):
        if ctb(b['ct']) != cfine or not b.get('name', '').startswith('inkeval'): continue
        r = subprocess.run([U, 'reg', 'map', '/vesuvius/ufsm/reg/xres/%s/reg.json' % reg, '--f2c', '--box', ','.join(map(str, b['l0']))], capture_output=True, text=True).stdout.split()
        l0 = [int(v) for v in r[-1].split(',')]; boxes.append(dict(name=b['name'] + ' (mapped to %s)' % ctb(rj['coarse']['key'])[:14], ct=rj['coarse']['key'], l0=l0, src='xres f2c ' + reg))
bycct = collections.defaultdict(list)
for n, s in src.items(): bycct[s['ct']].append(n)
slabs = []
def addslab(name_src, ct_full, a, t, note):
    sh = shape(name_src); ny, nx = (sh[1], sh[2]) if sh else (65536, 65536)
    slabs.append(dict(name='slab-%s-z%d' % (name_src.split('-')[0], a), ct=ct_full, l0=[a, 0, 0, t, ny, nx], src=note))
done = set()
# 0139 jointly
C, F = '20250728140407-9.362um-1.2m-113keV-masked.zarr', '20260102150214-2.399um-0.2m-78keV-masked.zarr'
idc = lambda a, b: (a, b); c2f = lambda a, b: ((a - 623) / 0.257 - 300, (b - 623) / 0.257 + 300)
g = [(n, idc) for n in bycct[C] if n.startswith('PHerc0139')] + [(n, c2f) for n in bycct[F] if n.startswith('PHerc0139')]
nzC = len(src['PHerc0139-gt']['h']) * 256; L, a, t = best_slab(g, 2048, nzC)
fa, fb = c2f(a, a + t); fa = max(0, int(fa)); fb = int(fb)
addslab('PHerc0139-gt', full['PHerc0139-gt']['ct'], a, t, 'joint 0139 slab (9.362 um)')
addslab('PHerc0139-u2ink', full['PHerc0139-u2ink']['ct'], fa, fb - fa, 'joint 0139 slab mapped to 2.399 um (xres affine, 300-row margin)')
for n, fn in g: src[n]['held'] += useg(src[n]['h'], *fn(a, a + t))
done |= {C, F}
for ct, names in bycct.items():
    if ct in done: continue
    names = [n for n in names if n != 'PHerc1218-b3' or True]
    if min(src[n]['held'] for n in names) >= 0.07: continue
    um = full[names[0]]['um']; tmin = 4096 if um < 3 else 2048
    nz = len(src[names[0]]['h']) * 256
    L, a, t = best_slab([(n, idc) for n in names], tmin, nz)
    addslab(names[0], full[names[0]]['ct'], a, t, 'slab for ' + ','.join(names))
    for n in names: src[n]['held'] += useg(src[n]['h'], a, a + t)
# 1299: the bottom-third box replaced by its slab (base.json already lacks it)
allb = boxes + slabs
for b in allb: b.setdefault('src', b.pop('from', 'existing'))
json.dump(dict(note=__doc__, boxes=allb), open(H + '/holdouts-20261114.json', 'w'), indent=1)
for n in sorted(src): print('%-22s %.3f' % (n, src[n]['held']))
for s in slabs: print(s)
print(len(allb), 'boxes')

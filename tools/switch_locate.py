#!/usr/bin/env python3
"""switch_locate.py CELLS.tsv [RAW_CELLS.tsv] --out PREFIX [--rows 16] [--deg 5] [--z0 384] [--step 4] [--slab 128] [--min-gt 8] [--err 0.5]

Sheet switches of an unroll's wraps from `ufsm wrap-repair --codes ... --cells` (per pair k -> k + 1 and cell of --rows wrap rows x --deg
degrees: intervals, GT turn counts gt0 / gt1 / gt2). A cell is wrong when it has >= --min-gt GT intervals and gt0 + gt2 >= --err of them:
gt2 = the pair spans two sheets (a sheet skipped between wraps k and k + 1), gt0 = both wraps on one sheet. Wrong cells of one pair and
one kind are joined into regions (4-neighbours, the angle circular). A wrap k that left its sheet over a region shows a gt0 region on one
side (pair k - 1 or k) and a gt2 region on the other: such pairs of overlapping regions are one SWITCH of wrap k (kind "switch"); a gt2
(gt0) region without an opposite partner within +-2 pairs is a turn the fit misses (adds) from that pair outward over the region
("skip" / "dup": the count). Cause hints per event:
  snap  : RAW_CELLS (the same check on the unsnapped wraps) has no wrong cells under the event (the snap made it);
  seam  : the event's z range starts and ends on slab boundaries (--slab native z) and spans <= 2 slabs;
  ref   : skip / dup events (an uncompensated count error: the integer reference or the fit's gauge over the region);
  fit   : switch events not explained by the snap or a seam (the local fit / fields).
Writes PREFIX.tsv (one row per event), PREFIX_cells.tsv (the marked wrap / cells per event) and prints a summary."""
import sys, csv, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('cells'); ap.add_argument('raw', nargs='?')
ap.add_argument('--out', required=True); ap.add_argument('--rows', type=int, default=16); ap.add_argument('--deg', type=float, default=5)
ap.add_argument('--z0', type=float, default=384); ap.add_argument('--step', type=float, default=4); ap.add_argument('--slab', type=float, default=128)
ap.add_argument('--min-gt', type=int, default=8); ap.add_argument('--err', type=float, default=0.5)
a = ap.parse_args()
NS = int(round(360 / a.deg))

def load(p):
    W = {}   # (k, rowcell, sector) -> 'dup' | 'skip'
    G = {}   # (k, rowcell, sector) -> (gt_n, gt0, gt2)
    for r in csv.DictReader(open(p), delimiter='\t'):
        k, rc, s = int(r['k']), int(r['row0']) // a.rows, int(r['sector'])
        gn, g0, g2 = int(r['gt_n']), int(r['gt0']), int(r['gt2'])
        G[(k, rc, s)] = (gn, g0, g2)
        if gn >= a.min_gt and g0 + g2 >= a.err * gn: W[(k, rc, s)] = 'dup' if g0 > g2 else 'skip'
    return W, G

W, G = load(a.cells)
Wr = load(a.raw)[0] if a.raw else None
# regions: connected wrong cells of one pair and kind
seen = set(); regions = []
for key, kind in W.items():
    if key in seen: continue
    comp = []; st = [key]; seen.add(key)
    while st:
        k, rc, s = st.pop(); comp.append((rc, s))
        for drc, ds in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (k, rc + drc, (s + ds) % NS)
            if nb in W and W[nb] == kind and nb not in seen: seen.add(nb); st.append(nb)
    regions.append({'k': key[0], 'kind': kind, 'cells': set(comp)})
# pair gt0 regions with gt2 regions of a neighbouring pair (|dk| <= 2) that overlap them
for i, r in enumerate(regions): r['id'] = i; r['partner'] = None
by_k = collections.defaultdict(list)
for r in regions: by_k[r['k']].append(r)
for r in sorted(regions, key=lambda r: -len(r['cells'])):
    if r['partner'] is not None or r['kind'] != 'skip': continue
    best, bo = None, 0
    for dk in (-1, 1, -2, 2):
        for q in by_k[r['k'] + dk]:
            if q['kind'] != 'dup' or q['partner'] is not None: continue
            o = len(r['cells'] & q['cells'])
            if o > bo: best, bo = q, o
    if best is not None and bo >= 0.25 * min(len(r['cells']), len(best['cells'])): r['partner'] = best['id']; best['partner'] = r['id']
events = []
for r in regions:
    if r['kind'] == 'dup' and r['partner'] is not None: continue   # (reported with its skip partner)
    cells = set(r['cells']); ks = [r['k']]
    if r['partner'] is not None: q = regions[r['partner']]; cells |= q['cells']; ks.append(q['k'])
    rcs = sorted(c[0] for c in cells); secs = sorted(set(c[1] for c in cells))
    # angle range (circular): the largest gap between occupied sectors is outside the range
    if len(secs) == NS: a0, a1 = -180.0, 180.0
    else:
        gaps = [((secs[(i + 1) % len(secs)] - secs[i]) % NS, i) for i in range(len(secs))]
        g, i = max(gaps); s0 = secs[(i + 1) % len(secs)]; s1 = secs[i]
        a0 = -180 + s0 * a.deg; a1 = -180 + (s1 + 1) * a.deg
    z0 = a.z0 + rcs[0] * a.rows * a.step; z1 = a.z0 + (rcs[-1] + 1) * a.rows * a.step
    gn = sum(G.get((k, c[0], c[1]), (0, 0, 0))[0] for k in ks for c in cells)
    ge = sum(G.get((k, c[0], c[1]), (0, 0, 0))[1] + G.get((k, c[0], c[1]), (0, 0, 0))[2] for k in ks for c in cells)
    kind = 'switch' if r['partner'] is not None else r['kind']
    wrap = (max(ks) if kind == 'switch' else r['k'] + 1)   # the wrap that left its sheet (switch) / the first wrap off the count (skip, dup)
    cause = 'fit' if kind == 'switch' else 'ref'
    if Wr is not None and not any((k, c[0], c[1]) in Wr for k in ks for c in cells): cause = 'snap'
    elif z0 % a.slab == 0 and z1 % a.slab == 0 and z1 - z0 <= 2 * a.slab: cause = 'seam'
    mw = list(range(min(ks) + 1, max(ks) + 1)) if kind == 'switch' else [r['k'] + 1]   # the wraps to mark (between the paired gaps)
    events.append(dict(wrap=wrap, pairs=','.join(str(k) for k in sorted(ks)), kind=kind, cause=cause, z0=z0, z1=z1, a0=a0, a1=a1,
                       cells=len(cells), gt_n=gn, gt_err=ge, cellset=cells, mwraps=mw))
events.sort(key=lambda e: -e['cells'])
with open(a.out + '.tsv', 'w') as f:
    cols = ['id', 'wrap', 'pairs', 'kind', 'cause', 'z0', 'z1', 'a0', 'a1', 'cells', 'gt_n', 'gt_err']
    f.write('\t'.join(cols) + '\n')
    for i, e in enumerate(events): e['id'] = i; f.write('\t'.join(str(e[c]) if not isinstance(e[c], float) else '%.0f' % e[c] for c in cols) + '\n')
with open(a.out + '_cells.tsv', 'w') as f:   # the marked (wrap, row cell, sector) of each event (thumbnails)
    f.write('id\twrap\trowcell\tsector\n')
    for e in events:
        for w in e['mwraps']:
            for rc, sc in sorted(e['cellset']): f.write('%d\t%d\t%d\t%d\n' % (e['id'], w, rc, sc))
cnt = collections.Counter((e['kind'], e['cause']) for e in events); cc = collections.Counter(e['kind'] for e in events)
big = [e for e in events if e['cells'] >= 4]
print('switch_locate: %d wrong cells -> %d regions -> %d events (%s); by kind x cause: %s; events >= 4 cells: %d' % (
    len(W), len(regions), len(events), dict(cc), dict(cnt), len(big)))
for e in events[:15]:
    print('  #%d w%03d pairs %s %s/%s z %.0f-%.0f angle %.0f..%.0f deg, %d cells, GT %d/%d wrong' % (e['id'], e['wrap'], e['pairs'], e['kind'], e['cause'], e['z0'], e['z1'], e['a0'], e['a1'], e['cells'], e['gt_err'], e['gt_n']))

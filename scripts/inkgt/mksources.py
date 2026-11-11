#!/usr/bin/env python3
"""scripts/inkgt/mksources.py D OUT_SOURCES OUT_HOLDOUTS: the human-ink sources fragment (one entry per labelled scan, heads ink only) and its
holdout file (wbench holdouts.json format: every wbench box, the 9.362 um ones also mapped into the 2.2-2.4 um scans, and the ink eval crops),
from D/segs.tsv, D/holdboxes.tsv and the splats D/gt/<scroll>.zarr (2026-11-11)."""
import json, os, sys
D, OS, OH = sys.argv[1:4]
M = 'https://dl.ash2txt.org/community-uploads/forrest/volcomp'
AX = {'PHercParis4': '/vesuvius/ufsm/teacher/axis-0.json', 'PHerc1667': '/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json',
      'PHerc0139': '/vesuvius/ufsm/wbench/xres/PHerc0139/axis.json', 'PHerc0814': '/vesuvius/ufsm/wbench/xres/PHerc0814/axis.json',
      'PHerc0500P2': '/vesuvius/ufsm/wbench/xres/PHerc0500P2/axis.json'}
scans = {}
for l in open(os.path.join(D, 'segs.tsv')):
    sc, nm, tx, lab, msk, key, um = l.rstrip('\n').split('\t'); scans.setdefault(sc, (key, float(um), []))[2].append(nm)
wb = json.load(open('/vesuvius/ufsm/wbench/holdouts.json'))
boxes = list(wb['boxes'])
for l in open(os.path.join(D, 'holdboxes.tsv')):
    sc, b, note = l.rstrip('\n').split('\t')
    if note.startswith('wbench') and 'mapped' not in note: continue   # (already in the wbench list, same CT)
    boxes.append({'name': note, 'scroll': sc, 'ct': scans[sc][0].replace('/volumes/', '/'), 'l0': [int(v) for v in b.split(',')]})
json.dump({'version': 'inkgt-20261111', 'note': 'level-0 voxels (z,y,x,nz,ny,nx) of the named CT; match a source by the CT basename; no training window may touch these '
           'boxes: all wbench boxes (holdouts.json wbench-20261025b), the 9.362 um ones of PHerc0139 / 0814 / 0500P2 also mapped into the 2.2-2.4 um scans '
           '(ufsm reg map --box, /vesuvius/ufsm/reg/xres), the 16 Paris 4 ink eval crops (wbench/inkeval/crops) and 8 PHerc0139 / 1667 ink eval crops '
           '(inkgt/evcrops); the human-ink splats are written with the same boxes --exclude\'d', 'boxes': boxes}, open(OH, 'w'), indent=1)
src = []
for sc, (key, um, segs) in sorted(scans.items()):
    z = os.path.join(D, 'gt', sc + '.zarr')
    if not os.path.exists(os.path.join(z, 'zarr.json')): print('no splat for', sc, file=sys.stderr); continue
    vox = {}; ink = no = 0
    for f in os.listdir(os.path.join(z, 'segments')):
        j = json.load(open(os.path.join(z, 'segments', f))); ink += j['ink_voxels']; no += j['noink_voxels']
    e = {'name': sc + '-inkgt', 'scope': 'global', 'root': M, 'ct': key, 'um': um, 'min_level': 1, 'max_level': 2, 'iso': 64, 'weight': 0,
         'holdout_file': OH, 'targets': {'ink': {'root': z, 'group': '.', 'encoding': 'prob254', 'min_level': 1}},
         'heads': {'phase': 0, 'normal': 0, 'density': 0, 'wind4': 0, 'rv': 0, 'ink': 1, 'fibre': 0},
         'note': 'human ink labels (HF scrollprize / open-data label_set 20260918, %d segments) ink-splat --face -6,0 at 4.8 um: 254 ink / 0 no ink on the '
                 'outermost ~14 um of papyrus at the open (recto) face, 255 unknown; %.0fM ink + %.0fM no-ink voxels at 4.8 um' % (len(segs), ink / 1e6, no / 1e6)}
    if sc in AX: e['axis'] = AX[sc]
    src.append(e)
tot = sum(float(e['note'].split('; ')[-1].split('M ink')[0]) + float(e['note'].split(' + ')[-1].split('M no-ink')[0]) for e in src)
for e in src:
    k = float(e['note'].split('; ')[-1].split('M ink')[0]) + float(e['note'].split(' + ')[-1].split('M no-ink')[0]); e['weight'] = round(k / tot, 4)
json.dump({'note': 'fragment for /vesuvius/ufsm/univ/sources-univ.json (workstream U4, agent inkB): weights by labelled voxels (sum 1, rescale on merge)',
           'sources': src}, open(OS, 'w'), indent=1)
for e in src: print(e['name'], e['weight'], e['note'].split('; ')[-1])

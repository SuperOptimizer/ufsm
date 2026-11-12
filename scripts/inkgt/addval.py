#!/usr/bin/env python3
"""scripts/inkgt/addval.py SEGS.json...: add each segment's published validation raster as "validation" (open data: validation.zarr/0
beside inklabels.zarr; HF: <seg>_validation_mask_v2.tif beside <seg>_inklabels*.tif), when its shape matches the labels'. inkfly then
holds those pixels out of the supervision (sources onfly "validation": "exclude", the default) or keeps only them ("only": the eval GT).
Rewrites the files in place (a .bak next to each)."""
import json, os, re, shutil, struct, sys

def zshape(p):
    try: return tuple(json.load(open(os.path.join(p, 'zarr.json')))['shape'])
    except Exception: return None

def tshape(p):   # (height, width) from the first IFD (classic or BigTIFF)
    try:
        with open(p, 'rb') as f:
            hd = f.read(16); e = '<' if hd[:2] == b'II' else '>'; big = struct.unpack(e + 'H', hd[2:4])[0] == 43
            off = struct.unpack(e + 'Q', hd[8:16])[0] if big else struct.unpack(e + 'I', hd[4:8])[0]; f.seek(off)
            n = struct.unpack(e + ('Q' if big else 'H'), f.read(8 if big else 2))[0]; esz = 20 if big else 12; w = h = None
            for _ in range(n):
                ent = f.read(esz); tag, typ = struct.unpack(e + 'HH', ent[:4]); val = ent[12:] if big else ent[8:]
                if tag not in (256, 257): continue
                v = struct.unpack(e + 'H', val[:2])[0] if typ == 3 else struct.unpack(e + 'I', val[:4])[0] if typ == 4 else struct.unpack(e + 'Q', val[:8])[0]
                if tag == 256: w = v
                elif tag == 257: h = v
            return (h, w)
    except Exception: return None

def val_of(lab):
    if lab.endswith('/inklabels.zarr/0'):
        v = lab[:-len('inklabels.zarr/0')] + 'validation.zarr/0'
        return (v, zshape(v), zshape(lab)) if os.path.isdir(v) else None
    m = re.match(r'(.*/)([^/]+?)_inklabels[^/]*\.tif$', lab)
    if m:
        v = m.group(1) + m.group(2) + '_validation_mask_v2.tif'
        return (v, tshape(v), tshape(lab)) if os.path.exists(v) else None
    return None

for fn in sys.argv[1:]:
    d = json.load(open(fn)); n = 0; bad = 0
    for e in d['segments']:
        r = val_of(e['labels'])
        if not r: e.pop('validation', None); continue
        v, vs, ls = r
        if vs is None or ls is None or tuple(vs) != tuple(ls): print(f'{fn}: {v}: shape {vs} != labels {ls}, skipped'); bad += 1; continue
        e['validation'] = v; n += 1
    shutil.copy(fn, fn + '.bak'); json.dump(d, open(fn, 'w'), indent=1)
    print(f'{fn}: {n} of {len(d["segments"])} label windows with a validation raster ({bad} shape mismatches)')

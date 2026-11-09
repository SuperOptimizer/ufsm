#!/usr/bin/env python3
"""Build deduplicated binary surface labels for every scroll with registered segments in the open-data bucket.

Per scroll (one CT scan each, see SCANS):
  1. select the registered segments that have a mesh on that scan (raw/ traces are excluded);
     named wraps (w012, w046-052, wrap07) first, newest first, and an older segment whose wraps are all already kept
     is dropped whole; then unnamed segments, newest first
  2. download their x/y/z.tif + meta.json (ETag-checked, shared cache)
  3. build/surface_dedup removes surface area within ~CELL_UM of an earlier (higher-priority) segment
  4. masked copies of the meshes (dropped grid points get z = -1) are rasterised into one binary store
     (ufsm raster --binary 1 --band-chamfer 4, 4.8 um labels for ~2.4 um scans, native labels otherwise)
  5. a held-out test box (~2.5 mm cube, densest labels) is chosen from a coarse label level
Writes OUT/<scroll>/{selection.json, dedup.tsv, labels.zarr, holdout.json, build.json} and OUT/scans.json.
usage: build_open_surfaces.py [--out DIR] [--scrolls A,B] [--threads 32]
"""
import argparse, json, re, shutil, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_surface_store import atomic_json, download, list_objects

REPO = Path(__file__).resolve().parents[1]
UFSM, DEDUP = REPO / 'build/ufsm', REPO / 'build/surface_dedup'
INV = '/vesuvius/ufsm/gt/open-data-inventory/inventory.json'
REMOTE = 'https://dl.ash2txt.org/community-uploads/forrest/volcomp'
LOCAL = Path('/vesuvius/usrm/volcomp')
AXES = Path('/vesuvius/ufsm/runs/m7-finetune/axes')
CELL_UM = 25.0          # dedup cell; a sample is a duplicate within one cell (25-50 um) of an earlier surface
HOLD_UM = 2458.0        # held-out cube edge (1024 voxels at 2.4 um)
# scroll -> scan whose registered meshes are rendered (finest volcomp scan >= 2.2 um with the most meshes)
SCANS = {'PHerc0009B': '20250820154339', 'PHerc0139': '20260102150214', 'PHerc0172': '20241024131839',
         'PHerc0343P': '20260304131111', 'PHerc0500P2': '20250526151718', 'PHerc0800': '20250521135224',
         'PHerc0814': '20260309142202', 'PHerc0841': '20260319124803', 'PHerc1447': '20250521151220',
         'PHerc1667': '20251217075048'}


def wraps(name):
    """wrap numbers named in a segment id: w012, w046-052, wrap07"""
    out = set()
    for a, b in re.findall(r'(?:^|[-_])w(?:rap)?(\d{2,3})(?:-(\d{2,3}))?(?=$|[-_])', name):
        out |= set(range(int(a), int(b or a) + 1))
    return out


def ct_for(scroll, vid):
    local = [p for p in (LOCAL / scroll).glob(f'{vid}-*.zarr')] if (LOCAL / scroll).exists() else []
    if local: return str(LOCAL), f'{scroll}/{local[0].name}'
    m7 = [r for r in json.load(open('/vesuvius/ufsm/runs/m7-finetune/m7_inventory.json')) if r['ct'].split('/')[-1].startswith(vid)]
    if not m7: raise ValueError(f'{scroll} {vid}: no volcomp CT')
    return REMOTE, m7[0]['ct']


def ct_shape(root, ct):
    p = f'{root}/{ct}/0/zarr.json'
    meta = json.loads(Path(p).read_text()) if not root.startswith('http') else json.loads(urllib.request.urlopen(p, timeout=60).read())
    return meta['shape']


def axis_for(scroll, ct):
    u = Path(f'/vesuvius/usrm/umbilicus/{scroll}/umbilicus-full-resolution.json')
    if u.exists() and str(LOCAL) in ct: return str(u)
    a = AXES / (ct.split('/')[-1].replace('.zarr', '') + '.json')
    return str(a) if a.exists() else None


def select(rows):
    named = sorted([r for r in rows if wraps(r['id'])], key=lambda r: r['id'], reverse=True)
    unnamed = sorted([r for r in rows if not wraps(r['id'])], key=lambda r: r['id'], reverse=True)
    kept, dropped, seen = [], [], set()
    for r in named:
        w = wraps(r['id'])
        if w <= seen: dropped.append(dict(id=r['id'], reason=f'wraps {sorted(w)} already kept from a newer segment')); continue
        seen |= w; kept.append(r)
    return kept + unnamed, dropped


def build(scroll, out, cache, threads):
    vid = SCANS[scroll]; d = out / scroll; d.mkdir(parents=True, exist_ok=True)
    inv = json.load(open(INV)); vols = inv['volumes']
    rows = [r for r in inv['segments'] if r['scroll'] == scroll and r['source'] == 'registered' and vid in r['tifxyz_on']]
    order, dropped = select(rows)
    um = float(re.search(r'-([0-9.]+)um\.tifxyz', order[0]['tifxyz_on'][vid]).group(1))
    root, ct = ct_for(scroll, vid); shape = ct_shape(root, ct)
    level = 1 if um < 4 else 0
    print(f'[{scroll}] scan {vid} {um} um {shape}, {len(rows)} meshes, {len(dropped)} dropped by wrap name', flush=True)
    # download
    objs = []
    for r in order:
        o, _ = list_objects(r['tifxyz_on'][vid], '/')
        objs += [x for x in o if x['key'].split('/')[-1] in ('x.tif', 'y.tif', 'z.tif', 'meta.json')]
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda o: download(o, cache), objs))
    meshes = [cache / 'aws' / r['tifxyz_on'][vid] for r in order]
    # dedup
    area = sum(r['area_cm2'] or 0 for r in order)
    caplog = int(np.clip(np.ceil(np.log2(max(area, 1) * 1e8 / CELL_UM ** 2 * 2 / 0.6)), 24, 33))
    masks = d / 'masks'; masks.mkdir(exist_ok=True)
    res = subprocess.run([str(DEDUP), '--cell', f'{CELL_UM / um:.4f}', '--out', str(masks), '--cap-log2', str(caplog), *map(str, meshes)],
                         capture_output=True, text=True, env={'OMP_NUM_THREADS': str(threads)})
    if res.returncode: raise RuntimeError(f'{scroll}: dedup failed: {res.stderr[-1500:]}')
    (d / 'dedup.tsv').write_text(res.stdout)
    stats = [l.split('\t') for l in res.stdout.strip().splitlines()]
    sel = []
    for r, s, m in zip(order, stats, meshes):
        valid, keep = int(s[4]), int(s[5]); frac = keep / max(valid, 1)
        e = dict(id=r['id'], mesh=str(m), area_cm2=r['area_cm2'], wraps=sorted(wraps(r['id'])), valid_points=valid, kept_points=keep, kept_fraction=round(frac, 4))
        if frac < 0.1: dropped.append(dict(e, reason='less than 10% unique surface')); continue
        dst = d / 'dedup' / r['id']
        if dst.exists(): shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in ('x.tif', 'y.tif', 'meta.json'): (dst / f).symlink_to(m / f)
        z = tifffile.imread(m / 'z.tif'); mk = np.fromfile(masks / f'{s[0]}.mask', np.uint8).reshape(z.shape)
        z = np.where(mk > 0, z, np.float32(-1)).astype(np.float32); tifffile.imwrite(dst / 'z.tif', z)
        sel.append(dict(e, tifxyz=str(dst)))
    atomic_json(d / 'selection.json', dict(scroll=scroll, scan=vid, um=um, kept=sel, dropped=dropped))
    print(f'[{scroll}] kept {len(sel)} of {len(rows)} (unique area ~{sum(e["area_cm2"] * e["kept_fraction"] for e in sel):.0f} of {area:.0f} cm2)', flush=True)
    # raster
    lab = d / 'labels.zarr'
    if lab.exists(): shutil.rmtree(lab)
    cmd = [str(UFSM), 'raster', str(lab), '--shape', ','.join(map(str, shape)), '--um', str(um), '--level', str(level), '--binary', '1',
           '--band-chamfer', '4', '--levels', '6', '--threads', str(threads), '--shard', '512', *[e['tifxyz'] for e in sel]]
    with open(d / 'raster.log', 'w') as log: subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    # held-out box from the label level 4 above the finest
    keys = sorted([p.name for p in lab.iterdir() if p.is_dir()], key=float)
    ck = keys[min(4, len(keys) - 1)]; f = 2 ** (level + keys.index(ck))
    cshape = json.loads((lab / ck / 'zarr.json').read_text())['shape']
    tmp = d / 'coarse.raw'
    subprocess.run([str(UFSM), 'read', str(lab), ck, '0', '0', '0', *map(str, cshape), str(tmp), '--threads', '16'], check=True, capture_output=True)
    occ = np.fromfile(tmp, np.uint8).reshape(cshape) > 0; tmp.unlink()
    edge = int(round(HOLD_UM / um / 128)) * 128; e = max(1, edge // f)
    dens = ndi.uniform_filter(occ.astype(np.float32), e, mode='constant')
    h = e // 2
    lim = [slice(h, max(h + 1, n - (e - h))) for n in cshape]
    sub = dens[tuple(lim)]; c = np.unravel_index(np.argmax(sub), sub.shape)
    o = [int(np.clip((c[i] + h - h) * f, 0, shape[i] - edge)) // 128 * 128 for i in range(3)]
    held = float(occ[tuple(slice(o[i] // f, (o[i] + edge) // f) for i in range(3))].sum() / max(occ.sum(), 1))
    box = [*o, edge, edge, edge]
    atomic_json(d / 'holdout.json', dict(box=box, label_fraction_held_out=round(held, 4), coarse_key=ck))
    info = dict(scroll=scroll, scan=vid, um=um, root=root, ct=ct, shape=shape, label_level=level, labels=str(lab),
                label_key=keys[0], axis=axis_for(scroll, ct), holdout=box, held_out_fraction=round(held, 4),
                segments=len(sel), command=cmd)
    atomic_json(d / 'build.json', info)
    print(f'[{scroll}] labels {lab} holdout {box} ({held:.1%} of label voxels)', flush=True)
    return info


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--out', type=Path, default=Path('/vesuvius/ufsm/gt/open-surfaces-20261007'))
    a.add_argument('--scrolls', default=''); a.add_argument('--threads', type=int, default=32)
    a.add_argument('--scan-map', default='', help='JSON {scroll: scan id} replacing SCANS (e.g. the ~9 um scans)')
    g = a.parse_args()
    if g.scan_map: SCANS.clear(); SCANS.update(json.loads(g.scan_map))
    g.scrolls = g.scrolls or ','.join(SCANS)
    g.out.mkdir(parents=True, exist_ok=True)
    done = json.loads((g.out / 'scans.json').read_text()) if (g.out / 'scans.json').exists() else {}
    for s in g.scrolls.split(','):
        try: done[s] = build(s, g.out, g.out, g.threads)
        except Exception as e: print(f'[{s}] FAILED: {e}', flush=True); done[s] = dict(error=str(e)[:2000])
        atomic_json(g.out / 'scans.json', done)


if __name__ == '__main__':
    main()

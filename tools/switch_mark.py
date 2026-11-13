#!/usr/bin/env python3
"""switch_mark.py RUN EVENTS_PREFIX OUT.jpg [--rows 16] [--deg 5] [--z0 384] [--step 4] [--fold 8] [--crops N --crop-dir DIR]

Marks the events of switch_locate.py (PREFIX_cells.tsv: event, wrap, row cell, sector) on the strip's long image
(RUN/strip/img_fill/strip_long.jpg, its CT half: 1 px = 5 flat grid px). Per stripe the flat grid is sampled every 5 px
(RUN/strip/strip/flat/stripes/<s>/{x,y,z,gu}.tif): wrap from gu (RUN/strip/strip/strip.tsv), z -> row cell, the angle about the
axis (RUN/axis.json, level-0 points / 4 = level-2 voxels) -> sector; marked pixels are tinted red (switch) / blue (skip / dup) with
an outline, event ids written at their centroid. OUT.jpg: the long strip folded into --fold rows, halved. --crops N: the N largest
events' crops (CT render around the event, unmarked and marked) at full long-image resolution into --crop-dir."""
import sys, csv, json, argparse, numpy as np, tifffile
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('prefix'); ap.add_argument('out')
ap.add_argument('--rows', type=int, default=16); ap.add_argument('--deg', type=float, default=5); ap.add_argument('--z0', type=float, default=384)
ap.add_argument('--step', type=float, default=4); ap.add_argument('--fold', type=int, default=8); ap.add_argument('--crops', type=int, default=0); ap.add_argument('--crop-dir')
ap.add_argument('--axis-scale', type=float, default=4)
a = ap.parse_args(); R = a.run; NS = int(round(360 / a.deg)); S = 5
ev = {int(r['id']): r for r in csv.DictReader(open(a.prefix + '.tsv'), delimiter='\t')}
look = {}
for r in csv.DictReader(open(a.prefix + '_cells.tsv'), delimiter='\t'): look[(int(r['wrap']), int(r['rowcell']), int(r['sector']))] = int(r['id'])
ks = np.array(list(look.keys()) or [(0, 0, 0)]); WMIN = int(ks[:, 0].min()); LA = np.full((int(ks[:, 0].max()) - WMIN + 1, int(ks[:, 1].max()) + 1, NS), -1, np.int32)
for (w, rc_, sc_), i in look.items(): LA[w - WMIN, rc_, sc_] = i
segs = [(int(r['col0']), int(r['wrap'])) for r in csv.DictReader(open(R + '/strip/strip/strip.tsv'), delimiter='\t')]
sc0 = np.array([s[0] for s in segs]); sw = np.array([s[1] for s in segs])
cp = json.load(open(R + '/axis.json'))['control_points']; cp.sort(key=lambda p: p['z'])
az = np.array([p['z'] for p in cp]) / a.axis_scale; ay = np.array([p['y'] for p in cp]) / a.axis_scale; ax_ = np.array([p['x'] for p in cp]) / a.axis_scale
BB = {}
longim = Image.open(R + '/strip/img_fill/strip_long.jpg').convert('RGB'); LW, LH2 = longim.size; LH = LH2 // 2
ct = np.array(longim.crop((0, 0, LW, LH))); evid = np.full((LH, LW), -1, np.int32)
for st in csv.DictReader(open(R + '/strip/strip/flat/stripes.tsv'), delimiter='\t'):
    d = R + '/strip/strip/flat/stripes/' + st['stripe']; oc0 = int(st['out_col0'])
    X = tifffile.imread(d + '/x.tif')[::S, ::S]; Y = tifffile.imread(d + '/y.tif')[::S, ::S]; Z = tifffile.imread(d + '/z.tif')[::S, ::S]; G = tifffile.imread(d + '/gu.tif')[::S, ::S]
    ok = (Z >= 0) & (G >= 0)
    wr = sw[np.clip(np.searchsorted(sc0, G, side='right') - 1, 0, len(sw) - 1)]
    cy = np.interp(Z, az, ay); cx = np.interp(Z, az, ax_); th = np.degrees(np.arctan2(Y - cy, X - cx))
    sec = (np.floor((th + 180) / a.deg).astype(int)) % NS; rc = np.floor((Z - a.z0) / (a.rows * a.step)).astype(int)
    wi = wr - WMIN; good = ok & (wi >= 0) & (wi < LA.shape[0]) & (rc >= 0) & (rc < LA.shape[1])
    e = np.full(Z.shape, -1, np.int32); e[good] = LA[wi[good], rc[good], sec[good]]
    x0 = oc0 // S; h, w = e.shape; h = min(h, LH); w = min(w, LW - x0)
    evid[:h, x0:x0 + w] = np.maximum(evid[:h, x0:x0 + w], e[:h, :w])
    for i in np.unique(e[e >= 0]):   # the event's level-2 bounding box (z, y, x) and pixel count
        mm = e == i; bb = BB.setdefault(int(i), [1e9, -1e9, 1e9, -1e9, 1e9, -1e9, 0])
        for j, V in enumerate((Z, Y, X)): bb[2 * j] = min(bb[2 * j], float(V[mm].min())); bb[2 * j + 1] = max(bb[2 * j + 1], float(V[mm].max()))
        bb[6] += int(mm.sum())
    print('switch_mark: %s marked %d px' % (st['stripe'], int((e >= 0).sum())), file=sys.stderr)
with open(a.out.replace('.jpg', '_boxes.tsv'), 'w') as f:   # area: marked px x (5 x 38.4 um)^2
    f.write('id\twrap\tkind\tcause\tz0\tz1\ty0\ty1\tx0\tx1\tarea_cm2\n')
    for i in sorted(BB): b = BB[i]; f.write('%d\t%s\t%s\t%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.3f\n' % (i, ev[i]['wrap'], ev[i]['kind'], ev[i]['cause'], *b[:6], b[6] * (5 * 0.00384) ** 2))
img = ct.copy().astype(np.float32); m = evid >= 0
kind = np.array([1 if ev[i]['kind'] == 'switch' else 2 for i in sorted(ev)] + [0]) if ev else np.zeros(1, int)
kk = np.zeros_like(evid); kk[m] = kind[evid[m]]
img[kk == 1] = img[kk == 1] * 0.5 + np.array([255, 0, 0]) * 0.5; img[kk == 2] = img[kk == 2] * 0.5 + np.array([0, 90, 255]) * 0.5
edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)); img[edge] = [255, 255, 0]
mk = Image.fromarray(img.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(mk)
cent = {}
for i in sorted(ev):
    ys, xs = np.nonzero(evid == i)
    if len(xs) >= 20: cent[i] = (int(xs.mean()), int(ys.mean()), int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())); dr.text((cent[i][0], cent[i][1]), str(i), fill=(255, 255, 255))
mk.save(a.out.replace('.jpg', '_long.jpg'), quality=88)
F = a.fold; seg = (LW + F - 1) // F; th_ = Image.new('RGB', (seg, F * (LH + 6)), (40, 40, 40))
for f in range(F): th_.paste(mk.crop((f * seg, 0, min(LW, (f + 1) * seg), LH)), (0, f * (LH + 6)))
th_ = th_.resize((seg // 2, F * (LH + 6) // 2)); th_.save(a.out, quality=88)
print('switch_mark: %d events placed, %d px marked; %s (%dx%d), %s' % (len(cent), int(m.sum()), a.out, th_.size[0], th_.size[1], a.out.replace('.jpg', '_long.jpg')))
if a.crops and a.crop_dir:
    import os; os.makedirs(a.crop_dir, exist_ok=True)
    for i in sorted(cent, key=lambda i: -int(ev[i]['cells']))[:a.crops]:
        cx_, cy_, x0, x1, y0, y1 = cent[i]; pad = 120; box = (max(0, x0 - pad), max(0, y0 - pad), min(LW, x1 + pad), min(LH, y1 + pad))
        both = Image.new('RGB', (box[2] - box[0], 2 * (box[3] - box[1]) + 8), (40, 40, 40))
        both.paste(Image.fromarray(ct).crop(box), (0, 0)); both.paste(mk.crop(box), (0, box[3] - box[1] + 8))
        both.save('%s/event%03d.jpg' % (a.crop_dir, i), quality=90)
    print('switch_mark: crops of the %d largest events -> %s' % (min(a.crops, len(cent)), a.crop_dir))

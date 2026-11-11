#!/bin/bash
# hpipe.sh CROPS: label windows of every human-labelled segment reaching each crop -> splat into hlab (merge) -> the crops at 2.4 and 1.129 um
cd /vesuvius/ufsm/pbr; CR=$1; W=${CR%.txt}.windows
nice python3 -I - $CR > $W <<'PY'
import sys, numpy as np, tifffile, json, glob, os
for line in open(sys.argv[1]):
    nm, sg, u0, v0, n = line.split(); u0, v0, n = int(u0), int(v0), int(n)
    d = 'seg/%s/mesh/tifxyz' % sg; X = tifffile.imread(d + '/x.tif'); Y = tifffile.imread(d + '/y.tif'); Z = tifffile.imread(d + '/z.tif')
    c0, r0, c1, r1 = u0 // 20, v0 // 20, (u0 + n) // 20 + 1, (v0 + n) // 20 + 1; x, y, z = X[r0:r1, c0:c1], Y[r0:r1, c0:c1], Z[r0:r1, c0:c1]; m = z >= 0
    lo = np.array([z[m].min(), y[m].min(), x[m].min()]) - 50; hi = np.array([z[m].max(), y[m].max(), x[m].max()]) + 50
    for w in sorted(glob.glob('hfink/*')):
        sc = json.load(open(w + '/meta.json'))['scale'][0]
        LX = tifffile.imread(w + '/x.tif'); LY = tifffile.imread(w + '/y.tif'); LZ = tifffile.imread(w + '/z.tif')
        mm = (LZ >= lo[0]) & (LZ < hi[0]) & (LY >= lo[1]) & (LY < hi[1]) & (LX >= lo[2]) & (LX < hi[2])
        if mm.sum() < 50: continue
        lab = [f for f in os.listdir(w) if f.endswith(('_inklabels.tif', '_inklabels.tiff')) and not f[0].isdigit()][0]
        msk = [f for f in os.listdir(w) if 'supervision_mask.tif' in f and not f[0].isdigit()][0]
        rr, cc = np.nonzero(mm)
        print(os.path.basename(w), nm, 'vertices %d' % mm.sum(), 'label window c0 %d r0 %d w %d h %d' % (cc.min() / sc, rr.min() / sc, (cc.max() + 1 - cc.min()) / sc, (rr.max() + 1 - rr.min()) / sc), lab, msk)
PY
sed -e "s#hlab_windows.txt#$W#" -e 's/echo splat done/echo splat3 done/' splat.sh > splat3.sh; bash splat3.sh > splat3.log 2>&1
sed -e "s#< hcrops.txt#< $CR#" -e 's/echo batchh2 done/echo hpipe done/' batchh2.sh > batchh3.sh; bash batchh3.sh

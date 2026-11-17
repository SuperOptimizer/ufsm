#!/usr/bin/env python3
"""scripts/pseudo/agreemask.py AGREE.tsv OUT.boxes [MIN 0.8] [MINPAIRS 50]: `ufsm surf-labels --gt OTHER --gt-cells` rows (z y x n cell
kept_gt_both kept_lt25 records pair_mm pair_acc pairs) -> phase-coh --box-scale lines "z y x n n n 0" for the cells where two independent
fits disagree: pair accuracy < MIN over >= MINPAIRS pairs, or the kept voxels' phase agreement (within 1/4 turn) < MIN over >= 1000
voxels. Cells without enough evidence are left to the other gates. Prints the shares."""
import sys, math
src, out = sys.argv[1:3]; MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8; MP = int(sys.argv[4]) if len(sys.argv) > 4 else 50
rows = bad = 0; vox = badv = 0
with open(out, 'w') as fo:
    for i, l in enumerate(open(src)):
        if i == 0: continue
        v = l.split('\t'); z, y, x, n = (int(t) for t in v[:4]); kk = int(v[5]); a25 = float(v[6]); pa = float(v[9]); np_ = int(v[10])
        rows += 1; vox += kk
        if (np_ >= MP and pa == pa and pa < MIN) or (kk >= 1000 and a25 == a25 and a25 < MIN):
            bad += 1; badv += kk; fo.write('%d %d %d %d %d %d 0\n' % (z, y, x, n, n, n))
print('agreemask: %d cells, %d disagreeing masked (%.4f of the compared voxels), MIN %.2f' % (rows, bad, badv / max(vox, 1), MIN))

#!/usr/bin/env python3
"""sweep_plot.py TABLE.tsv OUT.png: the compression sweep's inference curves (sweep_table.py's table): per checkpoint the paired
difference of p05x (top) and ids (bottom) to the uncompressed anchor against the rate, bytes per original voxel (log), for volcomp
(circles) and latent cells (squares), with the 95% bootstrap interval over boxes and the +-0.01 noise floor."""
import sys, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = list(csv.reader(open(sys.argv[1]), delimiter='\t')); H, R = R[0], R[1:]
tags = sorted({r[0] for r in R}, key=lambda t: {'u1': 0, 'wz896': 1, 'wind96': 2}.get(t, 9))
name = {'u1': 'u1-8000 (latent grid, universal)', 'wz896': 'lat-wz896 (latent grid, winding)', 'wind96': 'wind96 (voxel net)'}
C = {'volcomp': '#2a78d6', 'latvol': '#eb6834'}; M = {'volcomp': 'o', 'latvol': 's'}; INK, MUTED, SURF = '#0b0b0b', '#6b6a64', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c9c8c0', 'axes.linewidth': 0.8, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = plt.subplots(2, len(tags), figsize=(4.6 * len(tags), 6.6), sharex=True, sharey='row', facecolor=SURF)
for j, t in enumerate(tags):
    for i, (col, lab) in enumerate(((6, 'p05x'), (10, 'ids'))):
        a = ax[i][j]; a.set_facecolor(SURF); a.axhspan(-0.01, 0.01, color='#e4e3dc', lw=0, zorder=0); a.axhline(0, color='#9a998f', lw=0.8, zorder=1)
        a.axvline(1.0, color='#c9c8c0', lw=0.8, ls=':', zorder=1)
        for codec in ('volcomp', 'latvol'):
            P = sorted(((float(r[3]), float(r[col]), float(r[col + 1]), float(r[col + 2]), r[1]) for r in R if r[0] == t and r[2] == codec and int(r[4]) >= 12), reverse=True)
            if not P: continue
            x = [p[0] for p in P]; y = [p[1] for p in P]
            a.errorbar(x, y, yerr=[[p[1] - p[2] for p in P], [p[3] - p[1] for p in P]], color=C[codec], lw=1.6 if codec == 'volcomp' else 0, elinewidth=1, capsize=0, marker=M[codec], ms=5.5,
                       mec=SURF, mew=1, label={'volcomp': 'volcomp (quality ladder)', 'latvol': 'latent cells (latvol)'}[codec], zorder=3)
            for k, p in enumerate(P):   # direct labels: the setting of each point (latent: staggered, the points sit close in rate)
                s = ('q' + p[4][2:]) if codec == 'volcomp' else p[4][2:].replace('_', '/')
                a.annotate(s, (p[0], p[1]), textcoords='offset points', xytext=(0, 8 if codec == 'volcomp' else (-13, -22, -31)[k % 3]), ha='center', fontsize=6.5, color=MUTED)
        a.set_xscale('log'); a.grid(axis='y', color='#eceae3', lw=0.6); a.spines[['top', 'right']].set_visible(False)
        if i == 0: a.set_title(name.get(t, t), fontsize=10, loc='left', color=INK)
        if j == 0: a.set_ylabel(f'{lab}: difference to the uncompressed CT')
        if i == 1: a.set_xlabel('bytes per original voxel (uncompressed = 1, dotted)')
ax[0][0].legend(frameon=False, loc='lower right', fontsize=8)
ax[0][0].annotate('grey band: ±0.01 noise floor', (0.02, 0.04), xycoords='axes fraction', fontsize=7.5, color=MUTED)
fig.suptitle('How far can the CT be compressed before the winding nets lose accuracy? 14 held-out boxes, bars: 95% interval over boxes (latent labels: coefficients / step)', fontsize=10.5, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(sys.argv[2], dpi=150, facecolor=SURF)

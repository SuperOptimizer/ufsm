#!/usr/bin/env python3
"""scripts/univ/uqa_validate.py VALDIR TAG...: do the label-free unseen-block metrics (uqa.py) rank winding fits like the GT does?

VALDIR/<TAG>/uqa.tsv: uqa.py rows of TAG's kept wbench fits (one per core box); the GT: $WB/res/<TAG>.tsv (ids, p05x). Per metric:
  pooled    Spearman rho of metric vs GT over every (tag, box) pair (box difficulty dominates: mostly says whether the metric sees hard boxes)
  within    mean over boxes of the Spearman rho across the tags on that box (what the pairing uses: same box, different checkpoints)
  pairs     share of (box, tag pair) differences whose sign matches the GT difference's (|d GT| >= 0.02 only), oriented by the metric's
            better side (upair.py BETTER); 0.5 = no information
  ckpt      Spearman rho over the tags of the per-tag means (boxes all tags have)
GT metric: GT (env, default ids; p05x also printed). Clean boxes only (the wbench res rows; label-noise boxes are not in the core set).
"""
import itertools, os, sys
import numpy as np
from scipy.stats import spearmanr

WB = os.environ.get('WB', '/vesuvius/ufsm/wbench')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BETTER = {'h1': 1, 'on_pap': 1, 'gap_void': 1, 'contrast': 1, 'sp_cv': -1, 'jumps': -1, 'nrm_deg': -1, 'nrm30': -1, 'rv_agree': 1, 'ink_air': -1,
          'ink_band': 1, 'fib_chg_m': -1, 'slip_l1': -1, 'seams': -1, 'fit_phase': -1, 'fit_normal': -1, 'fit_density': -1}

def rows(p, key='box'):
    L = [l.rstrip('\n').split('\t') for l in open(p) if l.strip()]; h = L[0]
    return {r[h.index(key)]: dict(zip(h, r)) for r in L[1:] if r != h}

def f(x):
    try: v = float(x); return v if np.isfinite(v) else None
    except (ValueError, TypeError): return None

def main():
    vd, tags = sys.argv[1], sys.argv[2:]
    Q = {t: rows(os.path.join(vd, t, 'uqa.tsv')) for t in tags}; G = {t: rows(os.path.join(WB, 'res', t + '.tsv')) for t in tags}
    boxes = sorted(set.intersection(*[set(Q[t]) & set(G[t]) for t in tags])); print(f'{len(tags)} tags x {len(boxes)} boxes (every tag has them)')
    mets = [m for m in next(iter(Q[tags[0]].values())) if m not in ('box', 'ok', 'thr', 'phi', 'sp_um')]
    for gt in (os.environ.get('GT', 'ids'), 'p05x'):
        print(f'\n== vs GT {gt}\n{"metric":10s} {"better":>6s} {"pooled":>7s} {"within":>7s} {"pairs":>6s} {"n":>5s} {"ckpt":>6s}')
        for m in mets:
            s = BETTER.get(m, 1); P = [(f(Q[t][b][m]), f(G[t][b][gt])) for t in tags for b in boxes]; P = [(a, g) for a, g in P if a is not None and g is not None]
            if len(P) < 10: continue
            rp = spearmanr([a for a, _ in P], [g for _, g in P]).correlation
            wb, agree, n = [], 0, 0
            for b in boxes:
                v = [(f(Q[t][b][m]), f(G[t][b][gt])) for t in tags]; v = [x for x in v if x[0] is not None and x[1] is not None]
                if len(v) >= 4 and len({x[0] for x in v}) > 1 and len({x[1] for x in v}) > 1: wb.append(spearmanr(*zip(*v)).correlation)
                for (a1, g1), (a2, g2) in itertools.combinations(v, 2):
                    if abs(g2 - g1) < 0.02 or a2 == a1: continue
                    n += 1; agree += (np.sign(a2 - a1) * s == np.sign(g2 - g1))
            cm = [(np.mean([f(Q[t][b][m]) for b in boxes if f(Q[t][b][m]) is not None]), np.mean([f(G[t][b][gt]) for b in boxes])) for t in tags]
            rc = spearmanr(*zip(*cm)).correlation
            print(f'{m:10s} {"+" if s > 0 else "-":>6s} {rp * s:+7.3f} {np.mean(wb) * s if wb else np.nan:+7.3f} {agree / n if n else np.nan:6.3f} {n:5d} {rc * s:+6.3f}')
        print('(rho signed by the better side: + = the metric agrees with the GT)')

main()

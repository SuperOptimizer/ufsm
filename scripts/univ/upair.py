#!/usr/bin/env python3
"""scripts/univ/upair.py A/unseen.tsv B/unseen.tsv: two checkpoints' unseen-scroll rows (scripts/univ/ueval.sh) paired by box.

Per metric: A and B means over the boxes both have, mean(B - A) with a bootstrap 95 % CI (10000 resamples, seed 1) that resamples
scrolls, each drawn scroll bringing all its boxes (the boxes are 2-3 per scan, and boxes of one scan move together; resampling boxes
independently gave CIs too narrow, eval audit 11-14 #20; a column "scroll" is required, else boxes are resampled alone),
and on how many boxes B is better. The direction per metric (+ higher is better, - lower) is fixed below; '?' metrics have no clear
better side (sheet spacing is the scroll's own, the fit's raw loss depends on its fields) and get no win count. Few boxes (8) make wide
CIs: read a metric that moves the same way on most boxes and across metrics before reading a single mean.
"""
import random, sys

BETTER = {'h1': 1, 'on_pap': 1, 'gap_void': 1, 'contrast': 1, 'sp_cv': 0, 'jumps': -1, 'nrm_deg': 0, 'nrm30': 0, 'rv_agree': 1, 'ink_air': -1,
          'ink_band': 1, 'fib_chg_m': -1, 'slip_l1': -1, 'seams': -1, 'fit_phase': -1, 'fit_normal': -1, 'fit_density': -1}
# tiers from the validation against GT (scripts/univ/uqa_validate.py, 6 checkpoints x 44 wbench core boxes, 2026-11-12; DESIGN.md): the
# share of same-box checkpoint pairs whose metric difference has the GT p05x difference's sign. jumps 0.77 (ids 0.60): primary. h1 /
# gap_void / on_pap / contrast 0.61-0.64 (ids ~0.5): secondary. nrm_deg 0.25 (inverted: smoother normals = worse fits, over-smoothing),
# sp_cv 0.49, nrm30 0.59 but -0.09 pooled: info only (no win count). rv / ink / fibre / fit-residual metrics: not validated yet (no heads
# on the kept fits).
TIER = {'jumps': 'primary', 'h1': 'secondary', 'gap_void': 'secondary', 'on_pap': 'secondary', 'contrast': 'secondary', 'nrm_deg': 'info', 'sp_cv': 'info',
        'nrm30': 'info'}

def load(p):
    L = [l.rstrip('\n').split('\t') for l in open(p) if l.strip()]; h = L[0]; return h, {r[0]: dict(zip(h, r)) for r in L[1:]}

def num(x):
    try: return float(x)
    except ValueError: return None

def main():
    ha, A = load(sys.argv[1]); hb, Bm = load(sys.argv[2]); boxes = sorted(set(A) & set(Bm)); random.seed(1)
    print(f'{len(boxes)} boxes on {len(set(Bm[b].get("scroll", b) for b in boxes))} scrolls (CI: scrolls resampled): {" ".join(boxes)}')
    print(f'{"metric":12s} {"A":>9s} {"B":>9s} {"B-A":>9s} {"95% CI":>21s}  better  B wins  tier')
    for m in [h for h in hb if h in ha and h not in ('box', 'scroll', 'ok', 'thr')]:
        pr = [(num(A[b][m]), num(Bm[b][m])) for b in boxes]; pr = [(a, b) for a, b in pr if a is not None and b is not None]
        if not pr: continue
        d = [b - a for a, b in pr]; n = len(d); md = sum(d) / n
        keep = [bx for bx in boxes if num(A[bx][m]) is not None and num(Bm[bx][m]) is not None]
        grp = {}
        for bx, x in zip(keep, d): grp.setdefault(Bm[bx].get('scroll', bx), []).append(x)
        G = list(grp.values()); bs = []
        for _ in range(10000):
            pick = [random.choice(G) for _ in G]; tot = sum(len(g) for g in pick); bs.append(sum(sum(g) for g in pick) / tot)
        bs.sort(); lo, hi = bs[249], bs[9749]
        s = BETTER.get(m, 0); w = f'{sum(1 for x in d if x * s > 0)}/{n}' if s else '-'
        print(f'{m:12s} {sum(a for a, _ in pr) / n:9.4f} {sum(b for _, b in pr) / n:9.4f} {md:+9.4f} [{lo:+9.4f}, {hi:+9.4f}]  {"+" if s > 0 else "-" if s < 0 else "?":>6s}  {w:>6s}  {TIER.get(m, 'unvalidated')}')

main()

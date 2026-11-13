#!/usr/bin/env python3
"""scripts/univ/upair.py A/unseen.tsv B/unseen.tsv: two checkpoints' unseen-scroll rows (scripts/univ/ueval.sh) paired by box.

Per metric: A and B means over the boxes both have, mean(B - A) with a bootstrap 95 % CI over those boxes (10000 resamples, seed 1),
and on how many boxes B is better. The direction per metric (+ higher is better, - lower) is fixed below; '?' metrics have no clear
better side (sheet spacing is the scroll's own, the fit's raw loss depends on its fields) and get no win count. Few boxes (8) make wide
CIs: read a metric that moves the same way on most boxes and across metrics before reading a single mean.
"""
import random, sys

BETTER = {'h1': 1, 'on_pap': 1, 'gap_void': 1, 'contrast': 1, 'sp_cv': -1, 'jumps': -1, 'nrm_deg': -1, 'nrm30': -1, 'rv_agree': 1, 'ink_air': -1,
          'ink_band': 1, 'fib_chg_m': -1, 'slip_l1': -1, 'seams': -1, 'fit_phase': -1, 'fit_normal': -1, 'fit_density': -1}

def load(p):
    L = [l.rstrip('\n').split('\t') for l in open(p) if l.strip()]; h = L[0]; return h, {r[0]: dict(zip(h, r)) for r in L[1:]}

def num(x):
    try: return float(x)
    except ValueError: return None

def main():
    ha, A = load(sys.argv[1]); hb, Bm = load(sys.argv[2]); boxes = sorted(set(A) & set(Bm)); random.seed(1)
    print(f'{len(boxes)} boxes: {" ".join(boxes)}')
    print(f'{"metric":12s} {"A":>9s} {"B":>9s} {"B-A":>9s} {"95% CI":>21s}  better  B wins')
    for m in [h for h in hb if h in ha and h not in ('box', 'scroll', 'ok', 'thr')]:
        pr = [(num(A[b][m]), num(Bm[b][m])) for b in boxes]; pr = [(a, b) for a, b in pr if a is not None and b is not None]
        if not pr: continue
        d = [b - a for a, b in pr]; n = len(d); md = sum(d) / n
        bs = sorted(sum(random.choice(d) for _ in range(n)) / n for _ in range(10000)); lo, hi = bs[249], bs[9749]
        s = BETTER.get(m, 0); w = f'{sum(1 for x in d if x * s > 0)}/{n}' if s else '-'
        print(f'{m:12s} {sum(a for a, _ in pr) / n:9.4f} {sum(b for _, b in pr) / n:9.4f} {md:+9.4f} [{lo:+9.4f}, {hi:+9.4f}]  {"+" if s > 0 else "-" if s < 0 else "?":>6s}  {w:>6s}')

main()

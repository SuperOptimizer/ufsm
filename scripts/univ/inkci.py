#!/usr/bin/env python3
"""scripts/univ/inkci.py B/vscores.tsv [A/vscores.tsv]: ink-vscore rows with bootstrap 95 % CIs over label segments (eval audit 2026-11-14 #20).

A crop is <segment>_<k> (w00_0, PHerc0139_w040_1): crops of one segment share its labels and its papyrus, so they move together and the
segment is the resampling unit (10000 resamples, seed 1). Per file: the mean AUC, the mean best F1 and the mean best F1 above its trivial
all-ink floor 2 s / (1 + s) (s = the crop's ink share; a best F1 at threshold 0 is that floor, not skill). With A: the paired mean B - A
over the crops both have, its CI, and how many segments improve. vscores.tsv columns (ink-vscore): crop P N known ink_share auc thr f1best
f1_05 prec05 rec05.
"""
import random
import re
import sys


def load(p):
    R = {}
    for line in open(p):
        f = line.rstrip('\n').split('\t')
        if len(f) < 8:
            continue
        try:
            s, auc, f1 = float(f[4]), float(f[5]), float(f[7])
        except ValueError:
            continue
        if auc != auc:   # nan: a one-class crop
            continue
        R[f[0]] = {'auc': auc, 'f1': f1, 'f1x': f1 - 2 * s / (1 + s)}
    return R


def seg(c):
    return re.sub(r'_\d+$', '', c)


def boot(groups, rng, n=10000):
    G = list(groups.values())
    out = []
    for _ in range(n):
        pick = [rng.choice(G) for _ in G]
        out.append(sum(sum(g) for g in pick) / sum(len(g) for g in pick))
    out.sort()
    return out[n // 40], out[n - 1 - n // 40]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    B = load(sys.argv[1])
    A = load(sys.argv[2]) if len(sys.argv) > 2 else None
    rng = random.Random(1)
    crops = sorted(set(B) & set(A)) if A else sorted(B)
    if not crops:
        sys.exit('inkci: no crops')
    nseg = len({seg(c) for c in crops})
    print(f'ink CIs over {nseg} segments ({len(crops)} crops; segments resampled)')
    for m, nm in (('auc', 'AUC'), ('f1', 'best F1'), ('f1x', 'best F1 - floor')):
        gb = {}
        for c in crops:
            gb.setdefault(seg(c), []).append(B[c][m])
        lo, hi = boot(gb, rng)
        line = f'  {nm:16s} B {sum(B[c][m] for c in crops) / len(crops):.4f} [{lo:.4f}, {hi:.4f}]'
        if A:
            gd = {}
            for c in crops:
                gd.setdefault(seg(c), []).append(B[c][m] - A[c][m])
            dl, dh = boot(gd, rng)
            md = sum(B[c][m] - A[c][m] for c in crops) / len(crops)
            wins = sum(1 for g in gd.values() if sum(g) > 0)
            line += f'  A {sum(A[c][m] for c in crops) / len(crops):.4f}  B-A {md:+.4f} [{dl:+.4f}, {dh:+.4f}]  segments up {wins}/{len(gd)}'
        print(line)


if __name__ == '__main__':
    main()

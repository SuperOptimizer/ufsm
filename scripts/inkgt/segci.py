#!/usr/bin/env python3
"""scripts/inkgt/segci.py SCORES.TSV COL [COL ...]: the crop mean of each 1-based column with a 95 % bootstrap CI that resamples whole
segments (crop name <segment>_<k>: the crops of one segment are correlated; per-crop CIs are too narrow), 10000 resamples, seed 1.
Rows with a non-numeric value in a column are skipped for that column. Audit eval#20: the ink headlines had no CI at all, while the
CT-only baseline already spans AUC 0.46-0.56 per crop."""
import random, sys

def main():
    if len(sys.argv) < 3: sys.exit(__doc__)
    rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.strip()]
    for c in (int(x) for x in sys.argv[2:]):
        by = {}
        for r in rows:
            try: v = float(r[c - 1])
            except (ValueError, IndexError): continue
            if v != v: continue   # (NaN: a one-class crop has no AUC)
            by.setdefault(r[0].rsplit('_', 1)[0], []).append(v)
        G = list(by.values()); n = sum(len(g) for g in G)
        if not n: print(f'column {c}: no values'); continue
        m = sum(sum(g) for g in G) / n; random.seed(1); bs = []
        for _ in range(10000):
            pick = [random.choice(G) for _ in G]; bs.append(sum(sum(g) for g in pick) / sum(len(g) for g in pick))
        bs.sort()
        print(f'column {c}: mean {m:.4f} over {n} crops of {len(G)} segments, 95 % CI by segment [{bs[249]:.4f}, {bs[9749]:.4f}]')

main()

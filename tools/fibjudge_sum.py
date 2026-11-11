"""fibjudge_sum.py NAME rows.tsv...: pooled fibre-judge numbers of one scroll (our side only) and the tile spread (python3 -I)."""
import sys
import numpy as np
name = sys.argv[1]; rows = []
for p in sys.argv[2:]:
    L = open(p).read().splitlines()
    if len(L) < 2: continue
    h = L[0].split("\t"); rows += [dict(zip(h, l.split("\t"))) for l in L[1:]]
def pr(r, f, d): a, b = r[f"our_{f}_p{d}"].split(":"); return float(a), float(b)
out = [f"{name}: {len(rows)} tiles"]
for f in "HVU":
    A = sum(float(r[f"our_{f}_arc_m"]) for r in rows); S = sum(float(r[f"our_{f}_sw"]) for r in rows); N = sum(float(r[f"our_{f}_net"]) for r in rows)
    if A == 0: continue
    line = f"  {f}: arc {A:6.1f} m, excursions {S:6.0f} = {S / max(A, 1e-9):4.0f} /m, drift >= 0.5 turn per stretch {N / max(A, 1e-9):4.0f} /m; pairs within half a turn"
    for d in (16, 32, 64, 128, 256):
        n = sum(pr(r, f, d)[0] for r in rows); s = sum(pr(r, f, d)[1] for r in rows); line += f"  d{d} {s:.0f}/{n:.0f}={s / max(n, 1):.3f}"
    out.append(line)
    t = np.array([pr(r, f, 64)[1] / pr(r, f, 64)[0] for r in rows if pr(r, f, 64)[0] >= 100])
    if len(t): out.append(f"     tiles (>= 100 d64 pairs, n {len(t)}): d64 share min {t.min():.3f} p10 {np.percentile(t, 10):.3f} median {np.median(t):.3f} p90 {np.percentile(t, 90):.3f} max {t.max():.3f}")
print("\n".join(out))

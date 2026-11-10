"""fibjudge_stat.py rows.tsv...: per-box fibre scores vs the GT-based offset-cluster score (Spearman), pooled totals (python3 -I)."""
import sys
import numpy as np
from scipy.stats import spearmanr
rows = []; hdr = None
for p in sys.argv[1:]:
    for i, l in enumerate(open(p)):
        c = l.rstrip("\n").split("\t")
        if i == 0: hdr = c; continue
        rows.append(dict(zip(hdr, c)))
def pr(r, w, d):   # pooled H+V pair share within half a turn at distance d
    n = s = 0
    for f in "HV":
        a, b = r[f"{w}_{f}_p{d}"].split(":"); n += float(a); s += float(b)
    return n, s
def tot(r, w, k): return sum(float(r[f"{w}_{f}_{k}"]) for f in "HV")
print(f"{len(rows)} boxes")
for w in ("our", "gt"):
    A = sum(tot(r, w, "arc_m") for r in rows); S = sum(tot(r, w, "sw") for r in rows); N = sum(tot(r, w, "net") for r in rows)
    line = f"{w}: labelled arc {A:.1f} m, excursions {S:.0f} = {S / max(A, 1e-9):.0f} /m, persistent {N:.0f} = {N / max(A, 1e-9):.0f} /m; pairs within half a turn"
    for d in (16, 32, 64, 128, 256):
        n = sum(pr(r, w, d)[0] for r in rows); s = sum(pr(r, w, d)[1] for r in rows); line += f"  d{d}: {s:.0f}/{n:.0f} = {s / max(n, 1):.4f}"
    print(line)
    for f in "HV":
        A = sum(float(r[f"{w}_{f}_arc_m"]) for r in rows); S = sum(float(r[f"{w}_{f}_sw"]) for r in rows); N = sum(float(r[f"{w}_{f}_net"]) for r in rows)
        n, s = (sum(float(r[f"{w}_{f}_p64"].split(":")[k]) for r in rows) for k in (0, 1))
        print(f"   {f}: arc {A:.1f} m, excursions {S / max(A, 1e-9):.0f} /m, persistent {N / max(A, 1e-9):.0f} /m, d64 pairs {s:.0f}/{n:.0f} = {s / max(n, 1):.4f}")
ok = [r for r in rows if r["oc_n"] != "na" and float(r["oc_n"]) >= 3000 and tot(r, "our", "arc_m") >= 0.2 and pr(r, "our", 32)[0] >= 200]
print(f"\n{len(ok)} boxes with >= 3000 offset samples, >= 0.2 m of fibre arc on our labels, >= 200 d32 pairs")
if len(ok) >= 8:
    ids = np.array([float(r["oc_ids"]) for r in ok]); pa = np.array([float(r["oc_pair"]) for r in ok])
    M = {
        "our excursions /m": [tot(r, "our", "sw") / tot(r, "our", "arc_m") for r in ok],
        "our persistent /m": [tot(r, "our", "net") / tot(r, "our", "arc_m") for r in ok],
        "our d32 pair share": [pr(r, "our", 32)[1] / pr(r, "our", 32)[0] for r in ok],
        "our d64 pair share": [pr(r, "our", 64)[1] / max(1, pr(r, "our", 64)[0]) for r in ok],
        "gt excursions /m": [tot(r, "gt", "sw") / max(1e-9, tot(r, "gt", "arc_m")) for r in ok],
        "gt d32 pair share": [pr(r, "gt", 32)[1] / max(1, pr(r, "gt", 32)[0]) for r in ok],
        "our - gt d32 pair share": [pr(r, "our", 32)[1] / pr(r, "our", 32)[0] - pr(r, "gt", 32)[1] / max(1, pr(r, "gt", 32)[0]) for r in ok],
        "our known share": [float(r["our_known"]) for r in ok],
    }
    print(f"oc_ids: median {np.median(ids):.3f}, quartiles {np.percentile(ids, 25):.3f} {np.percentile(ids, 75):.3f}; oc_pair median {np.median(pa):.3f}")
    for k, v in M.items():
        v = np.array(v); a = spearmanr(v, ids); b = spearmanr(v, pa)
        print(f"  {k:26s} median {np.median(v):8.3f}  Spearman vs oc_ids {a.statistic:+.2f} (p {a.pvalue:.3g})  vs oc_pair {b.statistic:+.2f} (p {b.pvalue:.3g})")
    o = np.argsort(ids)
    print("  boxes by oc_ids (lowest 6 / highest 6): box, oc_ids, our exc/m, our d32, gt d32")
    for i in list(o[:6]) + list(o[-6:]):
        r = ok[i]; print(f"    {r['box']:32s} {ids[i]:.3f}  {M['our excursions /m'][i]:7.0f}  {M['our d32 pair share'][i]:.3f}  {M['gt d32 pair share'][i]:.3f}")

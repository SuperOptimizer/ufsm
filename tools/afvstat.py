"""afvstat.py FILE.afv [SAMPLE_EVERY]: read-only description of an AFV fibre collection (analysis only; run with python3 -I).
Prints the metadata (short), the fibre counts / lengths per family, the length distribution, the point spacing, the annotation keys
(sampled) and the bounds."""
import json, sqlite3, sys
import numpy as np

p = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 200
db = sqlite3.connect(f"file:{p}?mode=ro&immutable=1", uri=True)
md = dict(db.execute("select key, value from metadata"))
root = json.loads(md.get("root", "{}"))
vs = root.get("voxel_size_mm")
print("file", p)
print("space", root.get("vc_open_data_coordinate_space"), "voxel_mm", vs, "base_zyx", root.get("coordinate_base_shape_zyx"))
for k in ("source", "fiber_count", "point_count", "block_count", "block_segments", "geometry", "complete"):
    if k in md: print(" ", k, md[k][:200])
print("  metadata keys", sorted(md))
r = db.execute("select family, count(*), sum(length), sum(point_count), min(length), max(length) from fibers group by 1").fetchall()
tot_n = sum(x[1] for x in r); tot_l = sum(x[2] for x in r); tot_p = sum(x[3] for x in r)
for fam, n, l, pc, mn, mx in r:
    print(f"  family {fam}: {n} fibres, {l * vs / 1000:.1f} m, {pc} points, len {mn * vs:.3f}..{mx * vs:.2f} mm")
print(f"  total {tot_n} fibres, {tot_l * vs / 1000:.1f} m, {tot_p} points, mean segment {tot_l / max(1, tot_p - tot_n) * vs * 1000:.1f} um ({tot_l / max(1, tot_p - tot_n):.2f} native voxels)")
L = np.array([x[0] for x in db.execute("select length from fibers")]) * vs
q = np.percentile(L, [1, 5, 25, 50, 75, 95, 99])
print("  length mm percentiles 1/5/25/50/75/95/99:", " ".join(f"{v:.2f}" for v in q), "mean %.2f" % L.mean())
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)):
    m = (L >= lo) & (L < hi)
    print(f"    {lo:>4}-{hi if hi < 1e8 else 'inf':>4} mm: {m.sum():8d} fibres ({m.mean() * 100:5.1f} %), {L[m].sum() / 1000:8.1f} m ({L[m].sum() / L.sum() * 100:5.1f} %)")
b = db.execute("select min(min_x), max(max_x), min(min_y), max(max_y), min(min_z), max(max_z) from fibers").fetchone()
print("  bounds xyz", " ".join(f"{v:.0f}" for v in b))
# z profile of length (by fibre mid z), 16 bins
zz = np.array(db.execute("select (min_z + max_z) / 2, length from fibers").fetchall())
h, e = np.histogram(zz[:, 0], bins=16, weights=zz[:, 1] * vs / 1000)
print("  metres by mid-z bin:", " ".join(f"{e[i]:.0f}:{h[i]:.0f}" for i in range(16)))
# annotation keys
keys = {}; vals = {}; n = 0
for (a,) in db.execute(f"select annotation from fibers where id % {every} = 0"):
    try: j = json.loads(a)
    except Exception: keys["<bad json>"] = keys.get("<bad json>", 0) + 1; continue
    n += 1
    for k, v in j.items():
        keys[k] = keys.get(k, 0) + 1
        if k in ("classification_source", "source", "type", "hv_classification", "mergedSourceCount", "crop_fragment_count"):
            s = json.dumps(v)[:80]; d = vals.setdefault(k, {}); d[s] = d.get(s, 0) + 1
print(f"  annotation keys over {n} sampled fibres:", {k: v for k, v in sorted(keys.items())})
for k, d in vals.items():
    top = sorted(d.items(), key=lambda t: -t[1])[:6]
    print("   ", k, top)
row = db.execute("select annotation from fibers order by length desc limit 1").fetchone()
print("  longest fibre annotation:", row[0][:700])
nb = db.execute("select count(*), sum(length(points)) from blocks").fetchone()
print("  blocks", nb[0], "point bytes", nb[1], "-> points incl. shared", nb[1] // 24)

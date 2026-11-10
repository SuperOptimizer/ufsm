"""The public human fibre annotations (Hugging Face bucket scrollprize/datasets, spiral/<scroll>/fibers/*.json, type vc3d_fiber): summary
per scroll and conversion to UFIB (points level-0 z, y, x of the annotated scan). The files are untrusted data: json only. analysis only.
usage: hfjson.py DIR UM [OUT.fib]"""
import json, sys, os, glob, collections, struct
import numpy as np
d, um = sys.argv[1], float(sys.argv[2]); out = sys.argv[3] if len(sys.argv) > 3 else None
runs = []; c = collections.Counter(); L = collections.defaultdict(float); zr = [1e30, -1e30]; ncp = 0; scale = collections.Counter(); man = collections.Counter()
for f in sorted(glob.glob(d + "/*.json")):
    j = json.load(open(f)); c["files"] += 1; c["type:" + str(j.get("type"))] += 1
    for t in j.get("tags", []) or ["(none)"]: c["tag:" + t] += 1
    hv = j.get("hv_classification") or {}; fam = hv.get("manual_tag") or hv.get("automatic_tag") or "?"; c["manual:" + (hv.get("manual_tag") or "-")] += 1
    c["who:" + os.path.basename(f).split("_")[0]] += 1; ncp += len(j.get("control_points", [])); c["branches"] += len(j.get("branches", []))
    for cp in j.get("control_points", []):
        s = cp.get("segment_to_next") or {}; scale[s.get("trace_to_base_scale")] += 1; man[os.path.basename(str(s.get("fiber_manifest")))] += 1; c["mode:" + str(s.get("interp_mode"))] += 1
    p = np.array(j.get("line_points", []), dtype=np.float64)
    if len(p) < 2: c["no line"] += 1; continue
    l = np.linalg.norm(np.diff(p, axis=0), axis=1).sum(); L[fam[:1].upper()] += l; zr = [min(zr[0], p[:, 2].min()), max(zr[1], p[:, 2].max())]
    runs.append((fam[:1].upper() if fam[:1].upper() in "HV" else "?", p[:, ::-1].astype(np.float32)))
print(d, dict(c)); print("  control points", ncp, "trace_to_base_scale", dict(scale), "manifests", dict(man.most_common(4)))
print("  line length (m):", {k: round(v * um * 1e-6, 2) for k, v in L.items()}, "total", round(sum(L.values()) * um * 1e-6, 2), "z range (level-0 voxels)", [round(v) for v in zr],
      "point spacing median", round(float(np.median(np.concatenate([np.linalg.norm(np.diff(r[1], axis=0), axis=1) for r in runs]))), 2))
if out:
    off = np.cumsum([0] + [len(r[1]) for r in runs]).astype(np.int32)
    with open(out, "wb") as o:
        o.write(b"UFIB" + struct.pack("<iii", 1, len(runs), int(off[-1]))); o.write(off.tobytes()); o.write("".join(r[0] for r in runs).encode()); o.write(np.concatenate([r[1] for r in runs]).astype("<f4").tobytes())
    print("  ->", out)

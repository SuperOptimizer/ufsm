#!/usr/bin/env python3
"""scripts/inksmoke/mkarms.py H0.json TEACHDIR HOLDOUTS OUTDIR [ARMS (AB; A: before the teacher labels exist)]: the ink smoke's two arms (2026-11-18, lead-approved retrain plan).
H0: the human onfly ink sources (mksources_u2ink.py, LEVEL=1: the 4.8 / 9.6 um rungs). Both arms: every source's (and its onfly target's)
holdout_file = HOLDOUTS (holdouts-20261118: a superset of holdouts-ink for every trained scan), per-scroll weights capped (1667 was ~24 %
of the ink budget and the memorised scroll): WEIGHTS below, renormalised over the sources present.
  sources-A.json  human labels only
  sources-B.json  each scroll's segment list + its ink-teach windows (TEACHDIR/<SCROLL>/segs-teach.json: reader3 + ink9um, Platt per
                  scroll, gate p <= 0.1 | p >= 0.8, human px and held-out boxes removed), same weights (inkteach/ab/mkab.py's layout)"""
import json, os, sys
H0, TD, HO, OD = (os.path.abspath(a) for a in sys.argv[1:5]);   # absolute: the trainer opens the holdout files from its own cwd (pod 11-18)
ARMS = sys.argv[5] if len(sys.argv) > 5 else "AB"; os.makedirs(OD, exist_ok=True)
WEIGHTS = {"PHercParis4": 0.50, "PHerc0139": 0.20, "PHerc1667": 0.15, "PHerc0009B": 0.05, "PHerc0814": 0.05, "PHercMAN5": 0.05}
ONLY = [t for t in os.environ.get("SCROLLS", "").split(",") if t]   # podq.sh's 1-h smoke: a subset of the scrolls (weights renormalised)
src = json.load(open(H0)); HB = json.load(open(HO))
def holdfile(x, sc):   # HOLDOUTS limited to the boxes on this source's scan that touch one of its meshes' bboxes (+128 voxels): the ink
    # onfly path takes at most INKFLY_MAXEXCL (256; 64 before 11-18) boxes per scan; a box away from every
    # labelled mesh vertex (+128 voxels) never meets an ink label, so dropping it changes nothing.
    p = os.path.join(OD, f"holdouts-{sc}.json")
    if not os.path.exists(p):
        import numpy as np, tifffile
        ctb = os.path.basename(x["ct"]); pts = []
        for t in sorted({g["tifxyz"] for g in json.load(open(x["targets"]["ink"]["onfly"]["segments"]))["segments"]}):
            if not os.path.exists(os.path.join(t, "x.tif")): sys.exit(f"no {t}/x.tif: fetch the GT first")
            X, Y, Z = (tifffile.imread(os.path.join(t, a + ".tif")).astype(np.float32) for a in "xyz")
            v = (Z > 0) & (X > 0) & (Y > 0); pts.append(np.stack([Z[v], Y[v], X[v]], 1))   # (level-0 z, y, x of the mesh vertices; -1 = invalid)
        P = np.concatenate(pts); M = 128
        on = [b for b in HB["boxes"] if os.path.basename(b["ct"]) == ctb]
        keep = [b for b in on if np.any(np.all((P > np.array(b["l0"][:3]) - M) & (P < np.array(b["l0"][:3]) + np.array(b["l0"][3:]) + M), 1))]
        if len(keep) > 256: sys.exit(f"{sc}: {len(keep)} held-out boxes touch its meshes (the onfly limit, INKFLY_MAXEXCL, is 256)")
        json.dump({"note": f"{HO}: the {len(keep)} of {len(on)} boxes on {ctb} with a {sc} ink mesh vertex within 128 voxels (mkarms.py)", "boxes": keep}, open(p, "w"), indent=1)
        print(f"{sc}: {len(keep)} of {len(on)} held-out boxes touch its meshes -> {p}")
    return p
for arm in ARMS:
    s = json.loads(json.dumps(src)); keep = []
    for x in s["sources"]:
        sc = x["name"].split("-")[0]
        if sc not in WEIGHTS: print(f"{arm}: {x['name']} dropped (no weight)"); continue
        if ONLY and sc not in ONLY: print(f"{arm}: {x['name']} dropped (not in SCROLLS)"); continue
        x["weight"] = WEIGHTS[sc]; hf = holdfile(x, sc); x["holdout_file"] = hf; of = x["targets"]["ink"]["onfly"]; of["holdout_file"] = hf
        if arm == "B":
            t = os.path.join(TD, sc, "segs-teach.json")
            if not os.path.exists(t): sys.exit(f"B: no {t}")
            h = json.load(open(of["segments"])); tj = json.load(open(t)); nh = len(h["segments"])
            h["segments"] += tj["segments"]; h["note"] = "human windows + ink-teach windows: " + tj["note"]
            p = os.path.join(OD, f"segs-{sc}-HT.json"); json.dump(h, open(p, "w"), indent=1); of["segments"] = p; x["name"] = sc + "-u2inkT"
            print(f"B: {sc}: {nh} human + {len(tj['segments'])} teacher windows")
        keep.append(x)
    tot = sum(x["weight"] for x in keep)
    for x in keep: x["weight"] = round(x["weight"] / tot, 4)
    s["sources"] = keep; s["note"] = f"inksmoke arm {arm} (" + ("human only" if arm == "A" else "human + ink-teach") + f"), holdouts {HO}; " + s.get("note", "")
    json.dump(s, open(os.path.join(OD, f"sources-{arm}.json"), "w"), indent=1)
    print(f"{arm}: " + ", ".join(f"{x['name']} {x['weight']}" for x in keep))

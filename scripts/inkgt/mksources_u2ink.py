#!/usr/bin/env python3
"""scripts/inkgt/mksources_u2ink.py [IN] [OUT]: the U2 ink sources (2026-11-12): the human 3-D ink GT of the 8 scans as onfly ink targets at
level 0 (2.4 um; the sampler pools level 1, 4.8 um, from it: sources.c inkfly_open_pooled) instead of the level-1 onfly set
(IN, default /vesuvius/ufsm/inkgt/sources-inkfly.json). Per source: onfly "level" 0, min_level 0 / max_level 1 (the 2.4 and 4.8 um rungs:
train --um-rungs 2.4,4.8), "validation": "exclude" (the segments' validation_mask_v2 / validation.zarr pixels are held out, see
scripts/inkgt/addval.py), the ink head only (the winding heads 0: ink-only windows). Weights, holdout files and the face band as IN.
Env: ROOT (rewrites /vesuvius/ufsm/inkgt to ROOT in the sources, the holdout files and the segment lists, which are then written next to
OUT, whose directory is then the staging copy of ROOT: for a pod, ROOT=/workspace/univ/inkgt OUT=STAGE/inkgt/sources-u2ink.json),
CACHE_DIR (the onfly "cache_dir": the derived-chunk cache; default: the trainer's $UFSM_DERIVED_CACHE / ~/.cache/ufsm/derived),
CACHE_MB (each source's onfly memory cache; 2048 by default), CACHE (the sources' CT cache), LEVEL (0), VAL (exclude)."""
import json, os, shutil, sys

IN = sys.argv[1] if len(sys.argv) > 1 else "/vesuvius/ufsm/inkgt/sources-inkfly.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/vesuvius/ufsm/inkgt/sources-u2ink.json"
D = "/vesuvius/ufsm/inkgt"; ROOT = os.environ.get("ROOT", D).rstrip("/"); L = int(os.environ.get("LEVEL", "0")); VAL = os.environ.get("VAL", "exclude")
od = os.path.dirname(os.path.abspath(OUT))
def rw(p): return ROOT + p[len(D):] if ROOT != D and isinstance(p, str) and p.startswith(D) else p
def seglist(p):   # the segment list with rewritten paths (only when ROOT moves them)
    if ROOT == D: return p
    j = json.load(open(p))
    for s in j["segments"]:
        for k in ("tifxyz", "labels", "mask", "validation"):
            if k in s: s[k] = rw(s[k])
    json.dump(j, open(os.path.join(od, os.path.basename(p)), "w"), indent=1); return ROOT + "/" + os.path.basename(p)
S = json.load(open(IN)); out = []
for s in S["sources"]:
    s = json.loads(json.dumps(s)); ik = s["targets"]["ink"]; of = ik["onfly"]
    nv = sum("validation" in g for g in json.load(open(of["segments"]))["segments"])
    of["segments"] = seglist(of["segments"]); of["level"] = L; of["validation"] = VAL
    if os.environ.get("CACHE_DIR"): of["cache_dir"] = os.environ["CACHE_DIR"]
    if os.environ.get("CACHE_MB"): of["cache_mb"] = int(os.environ["CACHE_MB"])   # (the per-source memory cache, 2048 MB by default: 8 sources)
    if "holdout_file" in of: of["holdout_file"] = rw(of["holdout_file"])
    if "holdout_file" in s: s["holdout_file"] = rw(s["holdout_file"])
    if ROOT != D and isinstance(s.get("axis"), str) and not s["axis"].startswith(D):   # (desk axis files: copied next to OUT, axis/<source>.json)
        os.makedirs(os.path.join(od, "axis"), exist_ok=True); an = s["name"].replace("-inkfly", "") + ".json"
        shutil.copyfile(s["axis"], os.path.join(od, "axis", an)); s["axis"] = ROOT + "/axis/" + an
    ik["min_level"] = L; s["min_level"] = L; s["max_level"] = L + 1
    s["name"] = s["name"].replace("-inkfly", "-u2ink")
    s["heads"] = {k: (1 if k == "ink" else 0) for k in s.get("heads", {"ink": 1})}
    s["note"] = f"human ink GT onfly at level {L} (+ {L + 1} pooled), validation {VAL} ({nv} segments with a validation raster); mksources_u2ink.py"
    out.append(s)
J = {"cache": os.environ.get("CACHE", S.get("cache", "/vesuvius/ufsm/cache")), "note": f"U2 ink sources from {IN} (mksources_u2ink.py)", "sources": out}
if ROOT != D:   # the holdout file(s) next to OUT too (their boxes are CT voxels: unchanged)
    for hf in {s.get("holdout_file") for s in S["sources"]} - {None}:
        if hf.startswith(D): json.dump(json.load(open(hf)), open(os.path.join(od, os.path.basename(hf)), "w"), indent=1)
json.dump(J, open(OUT, "w"), indent=1)
for s in out: print(f"{s['name']:22s} um {s['um']:<6} levels {s['min_level']}-{s['max_level']} weight {s['weight']:.4f} {s['targets']['ink']['onfly']['segments']}")
print("->", OUT)

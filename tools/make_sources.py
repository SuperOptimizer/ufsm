#!/usr/bin/env python3
"""Generate a ufsm sources file from the exported ground truth under /vesuvius/ufsm/gt.

Sources:
  - HF label pyramids (gt/labels/<name>/) paired with the volcomp CT mirror of the same scan
  - rasterized segment labels (gt/raster/<scroll>/<vol>/) paired with that scan's CT mirror
  - Kaggle cubes (gt/kaggle/) as a shared-array regions source
CT comes from the local mirror under /vesuvius/usrm/volcomp when present, else the published tree over HTTPS
with the chunk cache. Usage: make_sources.py [--out configs/all.json] [--no-kaggle]
"""
import json, os, subprocess, sys, urllib.request, re

GT = "/vesuvius/ufsm/gt"
ROOT = "https://dl.ash2txt.org/community-uploads/forrest/volcomp"
LOCAL = "/vesuvius/usrm/volcomp"
UMB = "/vesuvius/usrm/umbilicus"
CACHE = "/vesuvius/ufsm/cache"

# HF label zarr -> (scroll, volume id, um)
HF = {
    "PHercMANBp-ct-2um_surface.zarr": ("PHercMANBp", "20251216152116", 2.399),
    "0500p2_5217.zarr": ("PHerc0500P2", "20250526151718", 2.215),
    "2.215um_0.4m_111keV_PHerc0343P_surface.zarr": ("PHerc0343P", "20260304131111", 2.215),
    "SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked_surface.zarr": ("PHerc1667", "20251217075048", 2.399),
    "s1_2.4um_gp.zarr": ("PHercParis4", "20260411134726", 2.4),
    "w023-w032.zarr": ("PHerc0139", "20260102150214", 2.399),
    "w037-w041.zarr": ("PHerc0139", "20260102150214", 2.399),
    "w047-w059.zarr": ("PHerc0139", "20260102150214", 2.399),
}


def listing(url):
    html = urllib.request.urlopen(url, timeout=60).read().decode()
    return re.findall(r'href="([^"]+)"', html)


def ct_for(scroll, vol):
    """(root, ct key) for a scan: local mirror if it has the volume, else the published tree."""
    loc = os.path.join(LOCAL, scroll)
    if os.path.isdir(loc):
        for d in os.listdir(loc):
            if d.startswith(vol) and d.endswith(".zarr") and os.path.exists(os.path.join(loc, d, "zarr.json")):
                return LOCAL, f"{scroll}/{d}"
    for d in listing(f"{ROOT}/{scroll}/volumes/"):
        if d.startswith(vol) and d.endswith(".zarr/"):
            return ROOT, f"{scroll}/volumes/{d.rstrip('/')}"
    return None, None


def holdout_for(label_dir, um):
    """The densest level-0 shard of an exported label pyramid, as a level-0 box [z,y,x,nz,ny,nx]."""
    try:
        lv = json.load(open(os.path.join(label_dir, "zarr.json")))["attributes"]["ome"]["multiscales"][0]["datasets"]
        um0 = lv[0]["coordinateTransformations"][0]["scale"][0]
        l0 = os.path.join(label_dir, lv[0]["path"])
        meta = json.load(open(os.path.join(l0, "zarr.json")))
        shard = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        best = None
        for z in os.listdir(os.path.join(l0, "c")):
            for y in os.listdir(os.path.join(l0, "c", z)):
                for x in os.listdir(os.path.join(l0, "c", z, y)):
                    sz = os.path.getsize(os.path.join(l0, "c", z, y, x))
                    if best is None or sz > best[0]:
                        best = (sz, int(z), int(y), int(x))
        if not best:
            return None
        f = int(round(um0 / um))          # level-0 voxels per label voxel (raster stores may start at level 1)
        return [best[1] * shard * f, best[2] * shard * f, best[3] * shard * f, shard * f, shard * f, shard * f]
    except Exception as e:
        print("holdout:", e, file=sys.stderr)
        return None


def axis_for(scroll):
    p = os.path.join(UMB, scroll, "umbilicus-full-resolution.json")
    if os.path.exists(p):
        return p
    p = os.path.join(GT, "axis", f"{scroll}.json")
    return p if os.path.exists(p) else None


def main():
    out = "configs/all.json"
    kaggle = True
    a = sys.argv[1:]
    if "--out" in a:
        out = a[a.index("--out") + 1]
    if "--no-kaggle" in a:
        kaggle = False
    srcs = []
    for name, (scroll, vol, um) in HF.items():
        d = os.path.join(GT, "labels", name)
        if not os.path.exists(os.path.join(d, "zarr.json")):
            print("missing", d, file=sys.stderr)
            continue
        root, ct = ct_for(scroll, vol)
        if not ct:
            print("no CT for", scroll, vol, file=sys.stderr)
            continue
        src = {"name": f"{scroll}-hf", "root": root, "ct": ct, "um": um,
               "targets": {"recto": {"root": d, "group": "."}}, "weight": 1.0, "trust_band": 8}
        h = holdout_for(d, um)
        if h:
            src["holdout"] = h
        srcs.append(src)
    hf_vols = {(sc, v) for sc, v, _ in HF.values()}
    rd = os.path.join(GT, "raster")
    if os.path.isdir(rd):
        for scroll in sorted(os.listdir(rd)):
            for vol in sorted(os.listdir(os.path.join(rd, scroll))):
                d = os.path.join(rd, scroll, vol)
                if not os.path.exists(os.path.join(d, "zarr.json")):
                    continue
                if (scroll, vol) in hf_vols:
                    continue   # the HF label zarr of this scan is the better source
                root, ct = ct_for(scroll, vol)
                if not ct:
                    continue
                lv = json.load(open(os.path.join(d, "zarr.json")))["attributes"]["ome"]["multiscales"][0]["datasets"]
                um = lv[0]["coordinateTransformations"][0]["scale"][0]
                # the raster pyramid may start at level 1 of the CT (um = 2 x native); record the CT's native um
                m = re.search(r"-([0-9.]+)um-", ct)
                ct_um = float(m.group(1)) if m else um
                src = {"name": f"{scroll}-seg", "root": root, "ct": ct, "um": ct_um,
                       "targets": {"recto": {"root": d, "group": "."}}, "weight": 0.5}
                h = holdout_for(d, ct_um)
                if h:
                    src["holdout"] = h
                srcs.append(src)
    if kaggle and os.path.exists(os.path.join(GT, "kaggle", "cubes.json")):
        cj = json.load(open(os.path.join(GT, "kaggle", "cubes.json")))
        srcs.append({"name": "kaggle", "root": os.path.join(GT, "kaggle"), "ct": "images.zarr", "um": 1.0,
                     "targets": {"recto": {"array": "labels.zarr", "size": cj["cube"], "origins": cj["origins"]}}, "weight": 1.0})
    for s in srcs:
        ax = axis_for(s["name"].split("-")[0])
        if ax:
            s["axis"] = ax
    cfg = {"cache": CACHE, "sources": srcs}
    # unique names (eval_holdouts keys its output directories by name): suffix repeats with the CT scan id
    seen = {}
    for s in srcs:
        seen.setdefault(s["name"], []).append(s)
    for name, group in seen.items():
        if len(group) > 1:
            for s in group:
                scan = os.path.basename(s["ct"]).split("-")[0]
                s["name"] = f"{name}-{scan}"
            names = [s["name"] for s in group]
            if len(set(names)) < len(names):   # same scan, different label sets: use the label directory instead
                for s in group:
                    lab = os.path.basename(os.path.normpath(s["targets"]["recto"]["root"])).replace(".zarr", "")
                    s["name"] = f"{name}-{lab}"
    for s in srcs:   # scans finer than 1.8 um: their levels 0-1 are finer than or equal to the rest of the set and enormous; train from level 2
        if s.get("um", 2.4) < 1.8 and "regions" not in json.dumps(s.get("targets", {})) and s["name"] != "kaggle":
            s["min_level"] = 2   # level 2 of a 1.1 um scan is 4.5 um, the other scans' level 1; its level 1 alone is ~100 GB of chunks
    json.dump(cfg, open(out, "w"), indent=1)
    print(f"{len(srcs)} sources -> {out}", file=sys.stderr)
    for s in srcs:
        print(f"  {s['name']:18s} {s['um']:6.3f} um  ct {s['root']}/{s['ct']}  axis={'yes' if 'axis' in s else 'no'}  holdout={s.get('holdout')}", file=sys.stderr)


if __name__ == "__main__":
    main()

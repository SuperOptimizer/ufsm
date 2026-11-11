#!/usr/bin/env bash
# scripts/fibers/mksources2.sh [OUT.json]: sources for fibhv-b from the ring labels (labels2, scripts/fibers/mklabels2.sh), usable while
# the rasters still run: fiber-raster writes shards in z-major order, so the z shard rows complete in order; every row not yet complete
# (all its ny x nx shards written, the CT being read for each) goes into a per-run holdout file (with the wbench holdouts), so no training
# window touches unwritten (all-unknown) labels. Rerun when more rows are done; the trainer reads the sources at (re)launch.
set -euo pipefail
O=${1:-/vesuvius/ufsm/fibers/sources-fibhv-b.json}; LAB=${LAB:-/vesuvius/ufsm/fibers/labels2}; HO=${O%.json}.holdouts.json
S=512; WBH=${WBH:-/vesuvius/ufsm/wbench/holdouts.json}
# (pod: CTROOT=<volcomp mirror> VOL=volumes/ AX0175=.. AX1447=.. WBH=.. CACHE=/workspace/cache LAB=/workspace/fibers/labels2)
CTROOT=${CTROOT:-/vesuvius/ufsm/v2/ct} VOL=${VOL:-} CACHE=${CACHE:-/vesuvius/ufsm/cache} AX0175=${AX0175:-/vesuvius/ufsm/pods/PHerc0175A/pod/scrolls/PHerc0175A/axis.json} \
AX1447=${AX1447:-/mnt/scratch/pods/PHerc1447/run/axis.json} python3 -I - "$LAB" "$O" "$HO" "$WBH" "$S" <<'PY'
import json, os, sys
lab, out, ho, wbh, S = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
E = os.environ; vol = E["VOL"]
ct = {"PHerc0175A": ("PHerc0175A/" + vol + "20250521115057-8.640um-1.2m-116keV-masked.zarr", E["AX0175"]),
      "PHerc1447": ("PHerc1447/" + vol + "20250521151220-8.640um-1.2m-116keV-masked.zarr", E["AX1447"])}
H = json.load(open(wbh)); boxes = list(H["boxes"]); src = []; tot = 0
for s, (key, ax) in ct.items():
    d = f"{lab}/{s}"; a = f"{d}/H/8.64"
    if not os.path.isdir(a): continue
    z = json.load(open(f"{a}/zarr.json")); sh = z["shape"]; at = z.get("attributes", {}).get("ufsm", {}); zb = at.get("zband", [0, sh[0]])
    ny, nx = -(-sh[1] // S), -(-sh[2] // S); r0, r1 = zb[0] // S, -(-zb[1] // S)
    done = os.path.exists(f"{d}/done"); full = []
    for r in range(r0, r1):
        n = sum(len(fs) for _, _, fs in os.walk(f"{a}/c/{r}")) if os.path.isdir(f"{a}/c/{r}") else 0
        nv = sum(len(fs) for _, _, fs in os.walk(f"{d}/V/8.64/c/{r}")) if os.path.isdir(f"{d}/V/8.64/c/{r}") else 0
        full.append(done or (n >= ny * nx and nv >= ny * nx))
    ok = [r0 + i for i, f in enumerate(full) if f]
    if not ok: continue
    # hold out every row that is not complete (contiguous runs)
    for i, f in enumerate(full):
        if not f: boxes.append({"name": f"{s}-unrastered-{r0 + i}", "scroll": s, "ct": key, "l0": [(r0 + i) * S, 0, 0, S, sh[1], sh[2]]})
    # and the z range outside the raster band
    if zb[0] > 0: boxes.append({"name": f"{s}-below-band", "scroll": s, "ct": key, "l0": [0, 0, 0, zb[0], sh[1], sh[2]]})
    if zb[1] < sh[0]: boxes.append({"name": f"{s}-above-band", "scroll": s, "ct": key, "l0": [zb[1], 0, 0, sh[0] - zb[1], sh[1], sh[2]]})
    w = len(ok) * ny * nx; tot += w
    src.append({"name": f"{s}-fib2", "scope": "local", "root": E["CTROOT"], "ct": key, "um": 8.64, "min_level": 0, "max_level": 0, "axis": ax, "iso": 64,
                "holdout_file": ho, "w": w,
                "targets": {"recto": {"root": f"{d}/H", "group": ".", "encoding": "prob254"}, "sheet": {"root": f"{d}/V", "group": ".", "encoding": "prob254"}}})
    print(f"{s}: {len(ok)} of {len(full)} shard rows complete ({'done' if done else 'rastering'}), rows {ok[0]}..{ok[-1]}")
for e in src: e["weight"] = e.pop("w") / tot
H2 = dict(H); H2["boxes"] = boxes; H2["note"] = H["note"] + "; plus the unrastered label rows of the fibhv-b sources (scripts/fibers/mksources2.sh)"
json.dump(H2, open(ho, "w"), indent=1); json.dump({"cache": E["CACHE"], "sources": src}, open(out, "w"), indent=1)
for e in src: print(e["name"], round(e["weight"], 3))
PY

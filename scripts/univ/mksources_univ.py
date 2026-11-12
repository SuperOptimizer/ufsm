"""scripts/univ/mksources_univ.py [OUT.json]: workstream B4, the universal net's sources (default /vesuvius/ufsm/univ/sources-univ.json).
Each entry carries its pyramid level(s) (a 2.4 um scan's level 1 = the 4.8 rung, level 2 = 9.6; the trainer's --um-rungs picks among
the levels a source reaches), a "heads" key (phase, normal, density, wind4 = winding; rv = recto / verso papyrus from the same band
phase), a weight, and its holdouts (the wbench holdouts.json: core boxes, Paris 4 HOLD = the fibre HOLD, rv eval boxes; plus each
template source's inline holdout list). Sources:
  - the 9.6 / 9.36 um winding GT of the latent winding runs (sources-wind-gt-nvme.json: Paris 4 June / July, PHerc0139, PHerc0500P2)
    and PHerc1667 (sources-wind-gt1667.json), unchanged;
  - B1's 4.8 um phase rasters (scripts/univ/b1_relabel.sh, b1_pod_p4A.sh) as level-1 sources, once their done flag exists:
    PHerc1667 A / B (whole scroll) and Paris 4 June A (only the packed 9.6 shards of /vesuvius/ufsm/univ/p4A48_shards.tsv, share of
    cells with sheet spacing < 150 um >= 0.625; windows elsewhere see unknown labels, so the sampler draws inside the stored chunks).
The 4.8 sets are separate entries at level 1 only (not level 1-2 of one entry): reading a 4.8 phase into a 9.6 window would need a
pooled read the sampler does not have; the 9.6 entries serve the 9.6 rung.
  - fibre (2026-11-11, plan approved by the coordinator): univA's fibre head reads targets "fibre_h" / "fibre_v" (prob254, 255 =
    unknown), enabled by heads {"fibre": W}. (a) The fibhv-d human / published fibre rasters (FIB/sources-fibhv-d.json, pulled from
    pod 1: recto = H -> fibre_h, sheet = V -> fibre_v) as fibre-only entries, their metre weights (floor 0.03 / 0.015) scaled to
    FIBSHARE (0.2) of all windows, fibhv-d's holdouts (wbench + Paris 4 HOLD + 1447-HOLD). (b) Teacher fibre rasters (fibhv-d
    predictions, gated by the HOLD check) at FIBT/<winding source name>/{H,V} with a done flag, added as fibre targets to that winding
    entry with heads fibre 0.3 (relative to the human / published fibre-only entries' 1).
Env: B1 (the B1 output root, /mnt/scratch/univ/b1), HEADS (JSON, default winding + rv on), FIB (/mnt/scratch/univ/fibre), FIBT
(FIB/teacher), FIBSHARE (0.2)."""
import copy, json, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else "/vesuvius/ufsm/univ/sources-univ.json"
if out.startswith("-"): sys.exit(__doc__)   # (no flags: the one argument is the output path)
B1 = os.environ.get("B1", "/mnt/scratch/univ/b1"); P4SH = os.environ.get("P4SH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "p4A48_shards.tsv")); MIRROR = os.environ.get("MIRROR", "https://dl.ash2txt.org/community-uploads/forrest/volcomp")
HEADS = json.loads(os.environ.get("HEADS", '{"phase": 1, "normal": 1, "density": 1, "wind4": 1, "rv": 1}'))
T = json.load(open("/mnt/scratch/lat/sources/sources-wind-gt-nvme.json")); T1667 = json.load(open("/mnt/scratch/lat/sources/sources-wind-gt1667.json"))
src = []
for s in T["sources"]: src.append(dict(copy.deepcopy(s), heads=HEADS))
g1667 = next(s for s in T1667["sources"] if s["name"] == "PHerc1667-gt"); src.append(dict(copy.deepcopy(g1667), heads=HEADS))
def lvl1(base, name, root, weight, note):
    e = copy.deepcopy(base); e.update(name=name, min_level=1, max_level=1, weight=weight, heads=HEADS, note=note)
    # level 1 (4.8 um) of the CT: the desk's local copies hold only levels >= 2 (1667: 2, 4, 5), so the mirror (cached; MIRROR=local path
    # once level 1 is mirrored with `ufsm mirror`)
    sc, scan = e["ct"].split("/", 1); e["root"] = MIRROR; e["ct"] = f"{sc}/volumes/{scan}" if MIRROR.startswith("http") else e["ct"]
    e["targets"]["band"]["phase"] = {"root": root, "key": "4.8", "level": 1}; return e
# On-the-fly phase (univA, master 2ea3a91; default since 2026-11-11, ONFLY=0 restores the stored sets): the 4.8 um winding entries read
# the phase from their level-1 codes in the sampler (pair rule, rmax 80 codes voxels = --radius 160 / --gs 2; codes decoded on the GPU,
# ~0.12 s per P 896 sample on the pod). 1667 A / B: byte-identical to the stored phase{A,B}48 (univA, 6 windows of 896^3). Paris 4
# June / July: codes{A,B}48 at level 1 over the whole scroll (the stored June set covered only 316 of the 549 packed 9.6 shards; with
# no storage cost the packed-region limit is gone); the July codes need the July segments + wind-assign rebuilt first. 0139 / 0500P2 /
# 1667 9.6 / p4full stay stored (fused A / B phase, not on the fly).
ONFLY = os.environ.get("ONFLY", "1") == "1"; C1667 = "/vesuvius/ufsm/gt/PHerc1667-wind-20261016"; CP4 = f"{B1}/gt/paris4"; CP4N = os.environ.get("CP4N", "/vesuvius/ufsm/univ/b1/gt/paris4")
def onfly(base, name, root, key, weight, note):
    e = lvl1(base, name, None, weight, note); e["targets"]["band"]["phase"] = {"onfly": {"root": root, "key": key, "rmax": 80}, "level": 1}; return e
for k in ("A", "B"):
    r = f"{B1}/gt/1667/phase{k}48"
    if ONFLY and os.path.exists(f"{C1667}/codes{k}/zarr.json"):
        src.append(onfly(g1667, f"PHerc1667-{k}48", f"{C1667}/codes{k}", "4.798", 0.1, f"1667 codes{k} 4.798, phase on the fly (pair rule, rmax 80)"))
    elif os.path.exists(f"{r}/done"):
        src.append(lvl1(g1667, f"PHerc1667-{k}48", r, 0.1, f"B1 4.8 um phase, 1667 codes{k} (phase-raster --gs 2 --pair 1 --radius 160)"))
# B3 (the all-spirals labels, good.sh gate): PHerc1299 (2026-11-12), only with B3=1 until its sampler-only band line is checked. Phase on
# the fly from the kept wraps' codes (rmax 20 = surf-labels --radius 20 on the 9.596 um grid) and ONE stored array, the phase weight
# (ufsm phase-coh --coh 0 on the gated phase, scripts/univ/b3_1299_mask.py): 0 in the held-out band, the direction-masked cells, air
# (CT < 64 with no CT >= 64 within 2 voxels), the gt-gate's removed cells and every voxel surf-labels ignored; 0.5 + 0.5 x the fibre
# factor in the fibre tiles; 1 elsewhere. The band also rejects whole windows (holdout_reject).
B3D = os.environ.get("B3D", "/vesuvius/ufsm/univ/b3")
if os.environ.get("B3", "0") == "1" and os.path.exists(f"{B3D}/PHerc1299/mask/weight.zarr/9.596/zarr.json") and os.path.exists(f"{B3D}/PHerc1299/codes.zarr/9.596/zarr.json"):
    src.append({"name": "PHerc1299-b3", "scope": "global", "root": "/vesuvius/ufsm/v2/ct", "ct": "PHerc1299/20260309130042-2.399um-0.2m-78keV-masked.zarr", "um": 2.399,
                "min_level": 2, "max_level": 2, "axis": "/mnt/scratch/pods/PHerc1299/run/axis.json", "iso": 64, "weight": float(os.environ.get("W1299", "0.1")), "heads": HEADS,
                "holdout_file": "/vesuvius/ufsm/univ/holdouts-1299.json", "holdout_reject": ["1299-zband"],
                "note": "B3 PHerc1299: our snapped wraps (71 of 88 kept), phase on the fly from codes (rmax 20), weight = band / direction / fibre / papyrus / gate mask",
                "targets": {"band": {"radius": 20, "span": 19, "phase": {"onfly": {"root": f"{B3D}/PHerc1299/codes.zarr", "key": "9.596", "rmax": 20}, "level": 2,
                                                                         "weight": {"root": f"{B3D}/PHerc1299/mask/weight.zarr", "key": "9.596"}}}}})
# PHerc1218 (2026-11-12): same form, no direction mask (aligned outward 0.9998, 27 reversed cells = 0.0002 of the pairs) and no fibre
# gate (the Paris 4 ramp is miscalibrated on the 8.64 um scans: latres's 1447 control). The weight = the gated phase's papyrus mask only
# (phase-coh --coh 0 --papyrus 64: 0 on air, gt-gate-removed and ignored voxels, 1 elsewhere). Codes and weight on /mnt/scratch
# (/vesuvius was down to 28 GB). Level 0 = 8.64 um, rmax 20 (surf-labels --radius 20 on that grid).
B3S = os.environ.get("B3S", "/mnt/scratch/univ/b3")
if os.environ.get("B3", "0") == "1" and os.path.exists(f"{B3S}/w1218/weight.zarr/8.64/zarr.json") and os.path.exists(f"{B3S}/labels/PHerc1218/codes.zarr/8.64/zarr.json"):
    src.append({"name": "PHerc1218-b3", "scope": "global", "root": "/vesuvius/ufsm/v2/ct", "ct": "PHerc1218/20250521120456-8.640um-1.2m-116keV-masked.zarr", "um": 8.64,
                "min_level": 0, "max_level": 0, "axis": "/mnt/scratch/pods/PHerc1218/unroll/axis.json", "iso": 64, "weight": float(os.environ.get("W1218", "0.1")), "heads": HEADS,
                "note": "B3 PHerc1218: our snapped wraps, gt-gate (0.179 removed), phase on the fly from codes (rmax 20), weight = gate / papyrus mask",
                "targets": {"band": {"radius": 20, "span": 19, "phase": {"onfly": {"root": f"{B3S}/labels/PHerc1218/codes.zarr", "key": "8.64", "rmax": 20}, "level": 0,
                                                                         "weight": {"root": f"{B3S}/w1218/weight.zarr", "key": "8.64"}}}}})
p4 = next(s for s in T["sources"] if s["name"] == "PHercParis4-june"); p4b = next(s for s in T["sources"] if s["name"] == "PHercParis4-july")
for k, base, nm, w in (("A", p4, "PHercParis4-june48", 0.2), ("B", p4b, "PHercParis4-july48", 0.12)):
    cr = next((d for d in (CP4, CP4N) if os.path.exists(f"{d}/codes{k}48/done")), None)   # July codes live on NVMe (b1_p4B48.sh, 2026-11-12)
    if ONFLY and cr:
        src.append(onfly(base, nm, f"{cr}/codes{k}48", "4.8", w, f"Paris 4 codes{k}48 at level 1, phase on the fly (pair rule, rmax 80), whole scroll"))
        continue
    r = f"{CP4}/phase{k}48"
    if k != "A" or not os.path.exists(f"{r}/4.8/zarr.json"): continue
    # stored + partial (2026-11-11): pod 2 was terminated at 280 of 339 boxes, ~500 shards of finished boxes still in flight. Every packed
    # 9.6 shard (2048^3 level-0 voxels) whose 8 4.8 um shards are not all on the desk is held out (after the wbench boxes, so the wbench
    # validation box stays first), and the weight scales with the complete share.
    sh = [tuple(map(int, l.split())) for l in open(P4SH) if l.strip()]
    def full(z, y, x): return all(os.path.exists(f"{r}/4.8/c/{2*z+a}/{2*y+b}/{2*x+c}") for a in (0, 1) for b in (0, 1) for c in (0, 1))
    miss = [s for s in sh if not full(*s)]; frac = 1 - len(miss) / len(sh)
    ho = json.load(open(p4["holdout_file"])); ct = next(b["ct"] for b in ho["boxes"] if b["ct"].startswith("PHercParis4/"))
    ho["boxes"] += [{"name": f"p4A48-missing-{z}-{y}-{x}", "scroll": "paris4", "ct": ct, "l0": [z*2048, y*2048, x*2048, 2048, 2048, 2048]} for z, y, x in miss]
    ho["note"] = ho.get("note", "") + f"; plus the {len(miss)} packed 9.6 shards of the Paris 4 June 4.8 phase set not complete on the desk (mksources_univ.py)"
    hf = os.path.join(os.path.dirname(out), "p4A48.holdouts.json"); json.dump(ho, open(hf, "w"), indent=1)
    e = lvl1(p4, nm, r, round(w * frac, 4), f"B1 4.8 um phase, June codes at level 1, PARTIAL: {len(sh)-len(miss)} of {len(sh)} packed shards complete, the rest held out ({hf})")
    e["holdout_file"] = hf; e["partial"] = True; src.append(e)
FIB = os.environ.get("FIB", "/mnt/scratch/univ/fibre"); FIBT = os.environ.get("FIBT", f"{FIB}/teacher"); FS = float(os.environ.get("FIBSHARE", "0.2"))
# The fibre-only entries with the planes computed on the fly from the polylines (univA, 2026-11-11: {"fibre": {afv, plane, radius,
# guard, ring, iso, extra, map}}, byte-identical to the stored fiber-raster output at every level), so the stored labels / labels2
# rasters (79 GB, lost with pod 1) are not rebuilt. The nine fibhv-d sets with the recipes of scripts/fibers/mklabels2.sh (the ring
# sets: radius 2, guard 2, ring 3, the all-lengths AFV as the veto, CT air negatives at iso 64) and mklabels3.sh (no air test: iso -1);
# their fibhv-d metre weights (floor 0.03 / 0.015) scaled to FIBSHARE, fibhv-d's holdouts. A set is skipped while its AFV is missing.
RAW = os.environ.get("FIBRAW", "/vesuvius/ufsm/fibers/raw"); FO = os.environ.get("FIBON", "/vesuvius/ufsm/fibers/onfly")
P4MAP = "/vesuvius/ufsm/reg/PParis4_2026to2023/reg.json"
FSETS = [  # name, scroll / scan, um, level, afv, extra, map, iso, fibhv-d weight
    ("PHerc1447-fib", "PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", 8.64, 0, f"{RAW}/PHerc1447_1mm_v2.afv", f"{RAW}/PHerc1447_all_v2.afv", None, 64, 0.2766, "1447"),
    ("PHerc0175A-fib", "PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr", 8.64, 0, f"{RAW}/PHerc0175A_1mm_v2.afv", f"{RAW}/0175A_SMALLMORE.afv", None, 64, 0.2140, "0175A"),
    ("PHercParis4-fib", "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr", 2.4, 2, f"{RAW}/Paris4_1mm_v2.afv", f"{RAW}/Paris4_all_v2.afv@map,{RAW}/Paris4_v1.afv", P4MAP, -1, 0.1927, "p4"),
    ("PHercParis4-spiral", "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr", 2.4, 2, f"{RAW}/spiral_fibers.afv", f"{RAW}/Paris4_v1.afv", None, -1, 0.1818, "p4"),
    ("PHerc0125-fib", "PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr", 9.362, 0, f"{RAW}/PHerc0125_3mm_v2.afv", "", None, -1, 0.03, "0125"),
    ("PHerc0139-hf", "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr", 2.399, 2, f"{FO}/PHerc0139.fib", "", None, -1, 0.03, "0139"),
    ("PHercParis4-hf", "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr", 2.4, 2, f"{FO}/PHercParis4.fib", "", None, -1, 0.03, "p4"),
    ("PHerc1451-hf", "PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr", 2.399, 2, f"{FO}/PHerc1451.fib", "", None, -1, 0.03, "1451"),
    ("PHerc0332-hf", "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr", 2.399, 2, f"{FO}/PHerc0332.fib", "", None, -1, 0.015, "0332")]
fsrc = []
for nm, ct, um, lv, afv, ex, mp, iso, w, ax in FSETS:
    need = [afv] + [x.split("@")[0] for x in ex.split(",") if x] + ([mp] if mp else [])
    import time   # a file still being downloaded (modified in the last 5 min, or a .bad checksum marker) counts as missing
    def ready(x): return os.path.exists(x) and not os.path.exists(x + ".bad") and time.time() - os.path.getmtime(x) > 300
    if not all(ready(x) for x in need): print(f"{nm}: skipped (missing or still downloading {[x for x in need if not ready(x)]})"); continue
    def plane(pl):
        f = {"afv": afv, "plane": pl, "radius": 2, "guard": 2, "ring": 3, "iso": iso}
        if ex: f["extra"] = ex
        if mp: f["map"] = mp
        return {"fibre": f}
    fsrc.append({"name": nm, "scope": "local", "root": MIRROR, "ct": ct, "um": um, "min_level": lv, "max_level": lv, "axis": f"{FO}/axis-{ax}.json",
                 "iso": 64, "holdout_file": f"{FO}/holdouts-fibre.json", "weight": w, "heads": {"fibre": 1},
                 "targets": {"fibre_h": plane("H"), "fibre_v": plane("V")}})
for s in src:   # teacher fibre targets on the winding entries
    d = f"{FIBT}/{s['name']}"
    if os.path.exists(f"{d}/done"):
        s["targets"]["fibre_h"] = {"root": f"{d}/H", "group": ".", "encoding": "prob254"}
        s["targets"]["fibre_v"] = {"root": f"{d}/V", "group": ".", "encoding": "prob254"}; s["heads"] = dict(s["heads"], fibre=0.3)
# 3-D ink GT (inkB, U4; INK = the fragment /vesuvius/ufsm/inkgt/sources-inkgt.json). Per the coordinator (2026-11-11): where a winding
# entry reads the same CT scan (Paris 4 20260411134726: june, july, june48; 1667 20251217075048: gt, A48, B48), the "ink" target goes
# onto that entry (heads ink 1), so one window trains winding + rv + ink; its holdouts become the union of the entry's and the ink file
# (the wbench boxes first, so the validation box stays first; plus the mapped 9.36 boxes and the ink eval crops). The other scans (0139
# and 0500P2, whose winding GT is on the 9.362 um scans, and 0814 / 0841 / 0009B / MAN5) stay ink-only entries, together INKSHARE of
# the windows (0.05; univA confirmed 2026-11-11 that ink-only windows train with the winding losses at 0), labelled-voxel weights floored at
# INKFLOOR (0.02 of the fragment).
INK = os.environ.get("INK", "/vesuvius/ufsm/inkgt/sources-inkgt.json"); IS = float(os.environ.get("INKSHARE", "0.05")); IF = float(os.environ.get("INKFLOOR", "0.02"))
def scan(ct): return ct.split("/")[-1]
def hunion(name, files):
    boxes, seen = [], set()
    for f in files:
        for bx in json.load(open(f))["boxes"]:
            k = (scan(bx["ct"]), tuple(bx["l0"]))
            if k not in seen: seen.add(k); boxes.append(bx)
    hf = os.path.join(os.path.dirname(os.path.abspath(out)), f"holdouts-{name}.json")
    json.dump({"note": "union of " + ", ".join(files) + " (mksources_univ.py)", "boxes": boxes}, open(hf, "w"), indent=1); return hf
isrc = []
for s in (json.load(open(INK))["sources"] if os.path.exists(INK) else []):
    host = [w for w in src if "band" in w["targets"] and scan(w["ct"]) == scan(s["ct"])]
    if not host: s["weight"] = max(s["weight"], IF); isrc.append(s); continue
    for w in host:
        w["targets"]["ink"] = copy.deepcopy(s["targets"]["ink"]); w["heads"] = dict(w["heads"], ink=1)
        w["holdout_file"] = hunion(w["name"], [w["holdout_file"], s["holdout_file"]])
if IS == 0: isrc = []
tot = sum(s["weight"] for s in src); ft = sum(s["weight"] for s in fsrc); it = sum(s["weight"] for s in isrc)
share = FS if fsrc else 0.0; ishare = IS if isrc else 0.0
for s in src: s["weight"] = round(s["weight"] / tot * (1 - share - ishare), 4)
for s in fsrc: s["weight"] = round(s["weight"] / ft * share, 4)
for s in isrc: s["weight"] = round(s["weight"] / it * ishare, 4)
src += fsrc + isrc
json.dump({"cache": T.get("cache", "/vesuvius/ufsm/cache"), "sources": src}, open(out, "w"), indent=1)
for s in src:
    tg = s["targets"]; ph = tg["band"]["phase"] if "band" in tg else None
    r = (ph.get("root") or "onfly:" + ph["onfly"]["root"]) if ph else (lambda v: v.get("root") or "onfly:" + v["fibre"]["afv"])(next(iter(tg.values())))
    print(f"{s['name']:22s} um {s['um']:<6} levels {s['min_level']}-{s['max_level']} weight {s['weight']:.3f} heads {','.join(k for k, v in s['heads'].items() if v)} {r}")
print("->", out)

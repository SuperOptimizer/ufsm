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
B1 = os.environ.get("B1", "/mnt/scratch/univ/b1"); MIRROR = os.environ.get("MIRROR", "https://dl.ash2txt.org/community-uploads/forrest/volcomp")
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
for k in ("A", "B"):
    r = f"{B1}/gt/1667/phase{k}48"
    if os.path.exists(f"{r}/done"):
        src.append(lvl1(g1667, f"PHerc1667-{k}48", r, 0.1, f"B1 4.8 um phase, 1667 codes{k} (phase-raster --gs 2 --pair 1 --radius 160)"))
p4 = next(s for s in T["sources"] if s["name"] == "PHercParis4-june"); r = f"{B1}/gt/paris4/phaseA48"
if os.path.exists(f"{r}/done") or os.path.exists(f"{r}/pulled"):
    src.append(lvl1(p4, "PHercParis4-june48", r, 0.2, "B1 4.8 um phase, June codes at level 1, packed shards only (/vesuvius/ufsm/univ/p4A48_shards.tsv)"))
FIB = os.environ.get("FIB", "/mnt/scratch/univ/fibre"); FIBT = os.environ.get("FIBT", f"{FIB}/teacher"); FS = float(os.environ.get("FIBSHARE", "0.2"))
fsrc = []
if os.path.exists(f"{FIB}/sources-fibhv-d.json"):
    pod = {"/workspace/fibd/labels": f"{FIB}/labels", "/workspace/fibers/labels2": f"{FIB}/labels2", "/workspace/fibers/meta": f"{FIB}/meta",
           "/workspace/fibd/sources-fibhv-d.holdouts.json": f"{FIB}/sources-fibhv-d.holdouts.json"}
    def loc(v):
        for a, b in pod.items():
            if isinstance(v, str) and v.startswith(a): return b + v[len(a):]
        return v
    for s in json.load(open(f"{FIB}/sources-fibhv-d.json"))["sources"]:
        tg = s.pop("targets"); h, v = loc(tg["recto"]["root"]), loc(tg["sheet"]["root"])
        if not (os.path.exists(f"{h}/zarr.json") and os.path.exists(f"{v}/zarr.json")): continue
        e = {k: loc(x) for k, x in s.items()}
        if e["root"].startswith("/workspace"): e["root"] = MIRROR   # pod-local CT copies (1447, 0175A) -> the mirror + cache
        e["targets"] = {"fibre_h": dict(tg["recto"], root=h), "fibre_v": dict(tg["sheet"], root=v)}; e["heads"] = {"fibre": 1}; fsrc.append(e)
for s in src:   # teacher fibre targets on the winding entries
    d = f"{FIBT}/{s['name']}"
    if os.path.exists(f"{d}/done"):
        s["targets"]["fibre_h"] = {"root": f"{d}/H", "group": ".", "encoding": "prob254"}
        s["targets"]["fibre_v"] = {"root": f"{d}/V", "group": ".", "encoding": "prob254"}; s["heads"] = dict(s["heads"], fibre=0.3)
tot = sum(s["weight"] for s in src); ft = sum(s["weight"] for s in fsrc); share = FS if fsrc else 0.0
for s in src: s["weight"] = round(s["weight"] / tot * (1 - share), 4)
for s in fsrc: s["weight"] = round(s["weight"] / ft * share, 4)
src += fsrc
json.dump({"cache": T.get("cache", "/vesuvius/ufsm/cache"), "sources": src}, open(out, "w"), indent=1)
for s in src:
    tg = s["targets"]; r = tg["band"]["phase"]["root"] if "band" in tg else tg["fibre_h"]["root"]
    print(f"{s['name']:22s} um {s['um']:<6} levels {s['min_level']}-{s['max_level']} weight {s['weight']:.3f} heads {','.join(s['heads'])} {r}")
print("->", out)

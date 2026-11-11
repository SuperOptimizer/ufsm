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
Env: B1 (the B1 output root, /mnt/scratch/univ/b1), HEADS (JSON, default winding + rv on)."""
import copy, json, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else "/vesuvius/ufsm/univ/sources-univ.json"
B1 = os.environ.get("B1", "/mnt/scratch/univ/b1")
HEADS = json.loads(os.environ.get("HEADS", '{"phase": 1, "normal": 1, "density": 1, "wind4": 1, "rv": 1}'))
T = json.load(open("/mnt/scratch/lat/sources/sources-wind-gt-nvme.json")); T1667 = json.load(open("/mnt/scratch/lat/sources/sources-wind-gt1667.json"))
src = []
for s in T["sources"]: src.append(dict(copy.deepcopy(s), heads=HEADS))
g1667 = next(s for s in T1667["sources"] if s["name"] == "PHerc1667-gt"); src.append(dict(copy.deepcopy(g1667), heads=HEADS))
def lvl1(base, name, root, weight, note):
    e = copy.deepcopy(base); e.update(name=name, min_level=1, max_level=1, weight=weight, heads=HEADS, note=note)
    e["targets"]["band"]["phase"] = {"root": root, "key": "4.8", "level": 1}; return e
for k in ("A", "B"):
    r = f"{B1}/gt/1667/phase{k}48"
    if os.path.exists(f"{r}/done"):
        src.append(lvl1(g1667, f"PHerc1667-{k}48", r, 0.1, f"B1 4.8 um phase, 1667 codes{k} (phase-raster --gs 2 --pair 1 --radius 160)"))
p4 = next(s for s in T["sources"] if s["name"] == "PHercParis4-june"); r = f"{B1}/gt/paris4/phaseA48"
if os.path.exists(f"{r}/done") or os.path.exists(f"{r}/pulled"):
    src.append(lvl1(p4, "PHercParis4-june48", r, 0.2, "B1 4.8 um phase, June codes at level 1, packed shards only (/vesuvius/ufsm/univ/p4A48_shards.tsv)"))
tot = sum(s["weight"] for s in src)
for s in src: s["weight"] = round(s["weight"] / tot, 4)
json.dump({"cache": T.get("cache", "/vesuvius/ufsm/cache"), "sources": src}, open(out, "w"), indent=1)
for s in src: print(f"{s['name']:22s} um {s['um']:<6} levels {s['min_level']}-{s['max_level']} weight {s['weight']:.3f} {s['targets']['band']['phase']['root']}")
print("->", out)

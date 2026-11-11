"""scripts/fibers/mksources3.py OUT.json: the fibhv-d sources (DESIGN.md 2026-11-11 "fibhv-d: every published fibre set") - the fibhv-b
ring labels of PHerc1447 / PHerc0175A (labels2, mklabels2.sh) plus the mklabels3.sh rasters (Paris 4 mapped, PHerc0125, the four human
fibre-line sets). Weights by traced fibre metres (published counts, DESIGN.md fibre table; human sets from tools/hfjson.py), with a floor
per source (FLOOR, default 0.03; half that for the 2.5 m PHerc0332 set). Holdouts = the wbench boxes (incl. Paris 4 HOLD) + 1447-HOLD
(l0 5120,3840,3072,512,768,768: the densest 1447_1mm_v2 cell, a cross-check box; fibhv-a..c trained on it, fibhv-d starts from fibhv-c).
Env: CTL (local CT root of the 8.64 um scans), CTM (the volcomp mirror), L2, L3 (label dirs), META (axis-*.json, holdouts.json), CACHE."""
import json, os, sys
E = os.environ; out = sys.argv[1]; ho = out[:-5] + ".holdouts.json"
CTL, CTM, L2, L3, META = E["CTL"], E["CTM"], E["L2"], E["L3"], E["META"]; FLOOR = float(E.get("FLOOR", "0.03"))
def tg(d): return {"recto": {"root": f"{d}/H", "group": ".", "encoding": "prob254"}, "sheet": {"root": f"{d}/V", "group": ".", "encoding": "prob254"}}
S = [  # name, CT root, CT key, um, level, axis, label dir, metres
    ("PHerc1447-fib2", CTL, "PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", 8.64, 0, "axis-1447.json", f"{L2}/PHerc1447", 5065),
    ("PHerc0175A-fib2", CTL, "PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr", 8.64, 0, "axis-0175A.json", f"{L2}/PHerc0175A", 3915),
    ("PHercParis4-fib3", CTM, "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr", 2.4, 2, "axis-p4.json", f"{L3}/PHercParis4-l2", 3528),
    ("PHercParis4-spiral", CTM, "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr", 2.4, 2, "axis-p4.json", f"{L3}/PHercParis4-spiral", 3332),
    ("PHerc0125-fib3", CTM, "PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr", 9.362, 0, "axis-0125.json", f"{L3}/PHerc0125", 197),
    ("PHerc0139-hf", CTM, "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr", 2.399, 2, "axis-0139.json", f"{L3}/hf-PHerc0139", 30.65),
    ("PHercParis4-hf", CTM, "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr", 2.4, 2, "axis-p4.json", f"{L3}/hf-PHercParis4", 12.92),
    ("PHerc1451-hf", CTM, "PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr", 2.399, 2, "axis-1451.json", f"{L3}/hf-PHerc1451", 7.17),
    ("PHerc0332-hf", CTM, "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr", 2.399, 2, "axis-0332.json", f"{L3}/hf-PHerc0332", 2.48),
]
S = [s for s in S if os.path.exists(f"{s[6]}/done")]   # only finished rasters (a later resume picks up new ones)
fl = {s[0]: (FLOOR / 2 if s[7] < 5 else FLOOR) for s in S}
tot = sum(s[7] for s in S); w = {s[0]: s[7] / tot for s in S}
low = {k for k in w if w[k] < fl[k]}; rest = 1 - sum(fl[k] for k in low); big = sum(w[k] for k in w if k not in low)
for k in w: w[k] = fl[k] if k in low else w[k] / big * rest
src = []
for n, root, key, um, lv, ax, d, m in S:
    src.append({"name": n, "scope": "local", "root": root, "ct": key, "um": um, "min_level": lv, "max_level": lv,
                "axis": f"{META}/{ax}", "iso": 64, "holdout_file": ho, "targets": tg(d), "weight": w[n]})
    print(f"{n:18s} {m:8.2f} m  weight {w[n]:.3f}  {d}")
H = json.load(open(f"{META}/holdouts.json"))
H["boxes"].append({"name": "1447-HOLD", "scroll": "PHerc1447", "ct": "PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr", "l0": [5120, 3840, 3072, 512, 768, 768]})
H["note"] = H.get("note", "") + "; plus 1447-HOLD (fibhv-d cross-check box, scripts/fibers/mksources3.py)"
json.dump(H, open(ho, "w"), indent=1); json.dump({"cache": E.get("CACHE", "/workspace/cache"), "sources": src}, open(out, "w"), indent=1)
print("->", out, ho)

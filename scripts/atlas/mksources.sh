#!/bin/bash
# scripts/atlas/mksources.sh BASE.json OUT.json: BASE's sources + one "meshes" onfly source per ATLAS scroll with kept patches (scripts/atlas/gate.sh)
# (2026-11-27, atlas). Each ATLAS source: level-0 CT of the patches' scan, phase on the fly from the kept patches' q.tif (pair rule, rmax ~384 um),
# heads rv + normal only (the recto / verso side and the sheet normal; the phase gauge comes from our old fits, so phase / density / wind4 stay
# off), scope local (draws near the patches), holdouts $HO in mask mode. Weights: $TOTAL (0.15) split by kept area, each capped at $CAP (0.03).
# Env: A (/vesuvius/ufsm/atlas), K (kept dir), HO (holdouts json), TOTAL, CAP, CT (/vesuvius/ufsm/v2/ct)
set -u
BASE=${1:?BASE.json}; OUT=${2:?OUT.json}
A=${A:-/vesuvius/ufsm/atlas}; K=${K:-$A/kept}; HO=${HO:-$A/holdouts-atlas.json}; TOTAL=${TOTAL:-0.15}; CAP=${CAP:-0.03}; CT=${CT:-/vesuvius/ufsm/v2/ct}
# per scroll: kept area (cm2), scan, um, axis (the gate's)
awk -F'\t' -v K=$K 'FILENAME == ARGV[1] { if (FNR > 1) { a[$1] = $9; sc[$1] = $3; um[$1] = $4 } next }
  FNR > 1 && $18 == "pass" { s = $2; ar[s] += a[$1]; scan[s] = sc[$1]; u[s] = um[$1] }
  END { for (s in ar) printf "%s\t%.3f\t%s\t%s\n", s, ar[s], scan[s], u[s] }' $A/surf/index.tsv $K/PHerc*.audit.tsv | sort > $K/sources.tsv
tot=$(awk '{ t += $2 } END { print t }' $K/sources.tsv)
jq -n --arg note "atlas-20261127: $(basename $BASE) + ATLAS-15k kept patches (scripts/atlas/gate.sh; CC BY-NC 4.0, measurement / validation use) as meshes onfly sources, heads rv + normal, total weight $TOTAL by kept area (cap $CAP), holdouts $HO" \
  --slurpfile base $BASE \
  --rawfile tsv $K/sources.tsv --arg K $K --arg HO $HO --arg CT $CT --argjson TOT $TOTAL --argjson CAP $CAP --argjson AREA $tot \
  --rawfile axes $K/axes.tsv '
  ($axes | split("\n") | map(select(length > 0) | split("\t")) | map({(.[0]): .[1]}) | add) as $ax
  | $base[0] as $b
  | $b + {note: ($note + " | base note: " + ($b.note // "")),
          sources: ($b.sources + ($tsv | split("\n") | map(select(length > 0) | split("\t")) | map(
            {name: (.[0] + "-atlas"), scope: "local", root: $CT, ct: (.[0] + "/" + .[2]), um: (.[3] | tonumber), min_level: 0, max_level: 0,
             axis: $ax[.[0]], iso: 64, weight: ([($TOT * (.[1] | tonumber) / $AREA), $CAP] | min),
             targets: {band: {radius: 20, span: 19, phase: {level: 0, onfly: {meshes: {dir: ($K + "/" + .[0]), value: "q.tif"}, rmax: (if (.[3] | tonumber) < 9 then 44 else 41 end)}}}},
             heads: {phase: 0, normal: 1, density: 0, wind4: 0, rv: 1},
             holdout_file: $HO, holdout_mode: "mask"})))}' > $OUT
echo "$OUT: $(jq '.sources | length' $OUT) sources, ATLAS $(wc -l < $K/sources.tsv) ($(printf '%.1f' $tot) cm2 kept)"

#!/usr/bin/env bash
# mksources.sh K OUT.json: the latent winding runs' human-GT sources (desk /mnt/scratch/lat/sources/sources-wind-gt-nvme.json: Paris 4 June
# 0.36 / July 0.20, PHerc0139 0.22, PHerc0500P2 0.04, all scope global, band radius 20 / span 19, wbench holdouts) for a datakit pod:
# CT over HTTP from the volcomp mirror (<scroll>/volumes/<scan>) with a local cache, labels from K/gt.
set -euo pipefail; K=$1; OUT=$2; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
src() { jq -nc --arg n $1 --arg ct $2 --argjson um $3 --argjson lv $4 --arg ax $5 --argjson w $6 --arg pr $7 --arg pk $8 --arg M $M --arg ho $K/small/holdouts.json \
  '{name: $n, scope: "global", root: $M, ct: $ct, um: $um, min_level: $lv, max_level: $lv, axis: $ax, iso: 64, weight: $w, holdout_file: $ho,
    targets: {band: {radius: 20, span: 19, phase: {root: $pr, key: $pk, level: $lv}}}}'; }
{ src PHercParis4-june PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr 2.4 2 $K/small/axis/paris4-axis-0.json 0.36 $K/gt/paris4/phaseA 9.6
  src PHercParis4-july PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr 2.4 2 $K/small/axis/paris4-axis-0.json 0.20 $K/gt/paris4/phaseB 9.6
  src PHerc0139-gt PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr 9.362 0 $K/small/axis/PHerc0139-unroll-axis.json 0.22 $K/gt/p0139full/phase 9.362
  src PHerc0500P2-gt PHerc0500P2/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr 9.362 0 $K/small/axis/PHerc0500P2-20250820143440.json 0.04 $K/gt/PHerc0500P2-wind/phase 9.362
} | jq -s --arg c $K/cache '{cache: $c, sources: .}' > $OUT
echo "$OUT: $(jq '.sources | length' $OUT) sources"

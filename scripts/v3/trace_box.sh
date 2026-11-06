#!/bin/bash
# trace_box.sh DIR PROB Z,Y,X,NZ,NY,NX AXIS AXIS_SCALE [GT SGN]: VC3D patch tracer on one box (no Python).
#   DIR   work dir (ridge.zarr, ngrid, paths/, logs/); PROB the sheet prob (.raw box or zarr root, PKEY=array key);
#   env:  U=ufsm binary, IMG=VC3D podman image, VS=voxel um (9.596), NSEED=64, PAR=6 (parallel growers, 2 CPUs each), RNG=2,
#         STEP=5 (step_size), GEN=400 (generations), EFFORT=10 (search_effort), FIELD=ridge|prob, EXTRA_JSON (extra seed.json keys,
#         e.g. '"min_area_cm": 0.002'), EXPAND=0 (rounds of expansion mode after the seeds, EXPN growers each, default 64)
#   GT given: ufsm patch-score (recall / precision / purity) -> DIR/score.txt
set -euo pipefail
D=$1; PROB=$2; BOX=$3; AX=$4; AXS=$5; GT=${6:-}; SGN=${7:-1}
U=${U:-/vesuvius/ufsm/v3/ps-src/build/ufsm}; IMG=${IMG:-localhost/vc-local:20260828}; VS=${VS:-9.596}
NSEED=${NSEED:-64}; PAR=${PAR:-6}; RNG=${RNG:-2}; STEP=${STEP:-5}; GEN=${GEN:-400}; EFFORT=${EFFORT:-10}; FIELD=${FIELD:-ridge}
EXPAND=${EXPAND:-0}; EXPN=${EXPN:-64}; EXTRA_JSON=${EXTRA_JSON:-}
mkdir -p "$D"; D=$(cd "$D" && pwd); cd "$D"
rm -rf ridge.zarr ngrid vccache paths logs; mkdir -p vccache paths logs
$U trace-prep --prob "$PROB" ${PKEY:+--prob-key $PKEY} --box "$BOX" --axis "$AX" --axis-scale "$AXS" --out "$D" --vs "$VS" --field "$FIELD" \
   --seeds "$NSEED" --rng "$RNG" --threads 8 | tee prep.txt
mk() {   # mk MODE FILE
  printf '{"mode": "%s", "step_size": %s, "generations": %s, "min_area_cm": 0.005, "thread_limit": 2, "voxelsize": %s, "cache_root": "%s/vccache", "tgt_overlap_count": 10, "search_effort": %s, "normal_grid_path": "%s/ngrid"%s}\n' \
    "$1" "$STEP" "$GEN" "$VS" "$D" "$EFFORT" "$D" "${EXTRA_JSON:+, $EXTRA_JSON}" > "$2"
}
mk seed seed.json; mk expansion expand.json
podman run --rm -v "$D:$D" "$IMG" vc_gen_normalgrids -i "$D/ridge.zarr" -o "$D/ngrid" --spiral-step 5 --grid-step 32 --preview-every 0 > ngrid.log 2>&1
t0=$(date +%s); i=0
while read -r x y z; do
  ( timeout 1800 podman run --rm --cpus 2 -e OMP_NUM_THREADS=2 -v "$D:$D" "$IMG" vc_grow_seg_from_seed -v "$D/ridge.zarr" -t "$D/paths" -p "$D/seed.json" -s "$x" "$y" "$z" > "logs/s$i.log" 2>&1 || true ) &
  i=$((i + 1)); while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 1; done
done < seeds.txt
wait
for r in $(seq 1 "$EXPAND"); do
  for j in $(seq 1 "$EXPN"); do
    ( timeout 1800 podman run --rm --cpus 2 -e OMP_NUM_THREADS=2 -v "$D:$D" "$IMG" vc_grow_seg_from_seed -v "$D/ridge.zarr" -t "$D/paths" -p "$D/expand.json" > "logs/e${r}_$j.log" 2>&1 || true ) &
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 1; done
  done
  wait
done
echo "trace: $(ls paths | wc -l) patches in $(( $(date +%s) - t0 )) s" | tee trace.txt
if [ -n "$GT" ]; then $U patch-score --paths "$D/paths" --gt "$GT" --box "$BOX" --axis "$AX" --axis-scale "$AXS" --sgn "$SGN" --tsv "$D/patches.tsv" | tee score.txt; fi

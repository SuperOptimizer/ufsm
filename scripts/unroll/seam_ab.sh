#!/usr/bin/env bash
# seam_ab.sh: post-pass GPU work for audit unroll #5 (fit fields from single windows, tile-border seams; FBLEND=1 in unroll.sh, default
# off). The 1667 u1 fit fields on a 512 x 2048 x 2048 sub-box (level 2), three arms: A 640/64 (production), B 1152/64, C two grids
# (640/64 at 0 and 256) + field-blend --tile. Per arm: the predict seconds and the tile-border step of phase and t0 against the
# interior median (seam_ab.py; production showed t0 x4-5 at the 256-cell borders). ~25 min on one 16 GB card. Writes only under $W.
# Env: U, CK (u1 snap), R K (CT; default the desk store, a pod: R=https://dl.ash2txt.org/community-uploads/forrest/volcomp
# K=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr), AX (axis json), BOX, W (out), GPU.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$REPO/build/ufsm}; CK=${CK:-/vesuvius/ufsm/v2/lat/runs/u1/snap/13000.ckpt}
R=${R:-/vesuvius/ufsm/v2/ct}; K=${K:-PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr}; AX=${AX:-/vesuvius/ufsm/unroll/PHerc1667-u1/axis.json}
W=${W:-/vesuvius/ufsm/tmp-seamab/$(date +%m%d-%H%M)}; BOX=${BOX:-4096,1024,1024,512,2048,2048}; GPU=${GPU:-0}; mkdir -p "$W"
for f in "$U" "$CK" "$AX"; do [ -e "$f" ] || { echo "seam_ab: missing $f"; exit 1; }; done
fp() {   # fp OUT WIN HALO GRID
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU "$U" predict "$CK" "$R" "$K" "$1.tmp" --um 2.399 --level 2 --box "$BOX" --window $2 --halo $3 --shard 512 --grid-origin $4,$4,$4 --ema 1 --q 8 \
    --threads ${PTH:-8} --gpu 0 --axis "$AX" --levels 1 --fit-zarr "$1" --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 --occupancy 5 > "$1.log" 2>&1 || { echo "predict $1 failed"; tail -3 "$1.log"; return 1; }
  rm -rf "$1.tmp"; echo "$(basename "$1"): window $2 halo $3 grid $4: $(( $(date +%s) - t0 )) s, $(grep -m1 -oE '[0-9]+ windows to run' "$1.log")"
}
[ -e "$W/A/fields.json" ] || fp "$W/A" 640 64 0 || exit 1
[ -e "$W/B/fields.json" ] || fp "$W/B" 1152 64 0 || exit 1
[ -e "$W/C1/fields.json" ] || fp "$W/C1" 640 64 256 || exit 1
if [ ! -e "$W/C/fields.json" ]; then t0=$(date +%s)
  "$U" field-blend "$W/C" "$W/A" "$W/C1" --tile 512,512,512 --grids 0,0,0:256,256,256 --threads 8 > "$W/C.log" 2>&1 || { echo "field-blend failed"; tail -3 "$W/C.log"; exit 1; }
  echo "C: field-blend $(( $(date +%s) - t0 )) s"; fi
for a in A B C; do for p in pc ps t0 w; do "$U" read "$W/$a/$p" . 0 0 0 256 1024 1024 "$W/$a.$p.raw" --threads 8 > /dev/null 2>&1 || { echo "read $a $p failed"; exit 1; }; done; done
python3 -I "$REPO/scripts/unroll/seam_ab.py" "$W" $(( $(echo "$BOX" | cut -d, -f2) / 2 ))

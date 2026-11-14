#!/usr/bin/env bash
# scripts/infer/seam_ab.sh CKPT TAG: window / halo / seam-blend arms for a 6-level latent net (audit 2026-11-14 infer #3: the default halo is an
# alignment artefact, 16 cells against a 439-cell RF, and the bottleneck attention sees 125 .. 2197 tokens against training's 216).
# One PHerc1667 level-2 box, heads rv,ink,c0, every arm through ink-seam (step at the window borders, period = the interior) and z3-diff
# against arm B (the training token count):
#   A 640 / 64 (the default; 5^3 tokens)   B 768 / 128 (6^3 = training's at P 768)   C 896 / 192 (wider halo, 7^3)
#   D 1152 / 64 (k 2 windows, 9^3)         E 1152 / 64 --seam-inproc 1 (two grids blended; fields from grid 0: the ONEPASS recipe)
# The recipe change it decides is flags only (predict --window / --halo / --seam-inproc; unroll LATWROUND / ONEPASS): no default changes here.
# Env: GPU (0), U (build/ufsm), CTROOT, KEY, BOX (z,y,x,nz,ny,nx at level 2; default 1024 x 2048^2 around gn_ab's probe, on the 512 shard grid: every arm runs --full-grid 1, so the window borders sit at multiples of the interior from 0 and the arms compare shard for shard), AXIS (1667 axis json),
#      E (/vesuvius/ufsm/wbench/univ), WB (cache root), ARMS ("A B C D E"). Outputs $E/seamab-TAG/<arm>/<head>, summary $E/seamab-TAG/seam.txt.
# Cost (est.): ~10-20 min an arm at scratch2 size on one desk card; ~4-8 GB of lossless outputs an arm (delete after reading).
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}; H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}
R=${CTROOT:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}; K=${KEY:-PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr}
B=${BOX:-4096,512,1024,1024,2048,2048}; E=${E:-/vesuvius/ufsm/wbench/univ}; O=$E/seamab-$T; mkdir -p "$O"
AXA=(--no-axis 1); [ -n "${AXIS:-}" ] && AXA=(--axis "$AXIS")
arm() { case $1 in A) echo "640 64";; B) echo "768 128";; C) echo "896 192";; D|E) echo "1152 64";; esac; }
for a in ${ARMS:-A B C D E}; do
  set -- $(arm $a); w=$1; h=$2; X=(--full-grid 1); [ $a = E ] && X+=(--seam-inproc 1)
  [ -e "$O/$a/rv/zarr.json" ] && continue
  echo "$(date) arm $a: window $w halo $h ${X[*]}"
  $U predict "$CK" "$R" "$K" "$O/$a" --um 2.399 --level 2 --box $B --heads rv,ink,c0 --window $w --halo $h --shard 512 --q 0 --levels 1 \
     --gpu ${GPU:-0} --threads 8 --cache ${WB:-/vesuvius/ufsm/wbench}/cache "${AXA[@]}" "${X[@]}" > "$O/$a.log" 2>&1 || echo "seam_ab: arm $a failed ($O/$a.log)"
done
{ echo "seam_ab $T: box $B (level 2), shard 512"
  for a in ${ARMS:-A B C D E}; do set -- $(arm $a); st=$(($1 - 2 * $2)); [ -e "$O/$a/rv/zarr.json" ] || { echo "arm $a: missing"; continue; }
    echo "arm $a ($1 / $2, interior $st): $(grep -Eo '[0-9.]+ Mvox/s' "$O/$a.log" | tail -1)"
    for h in rv ink c0; do echo "  seam $h: $($U ink-seam "$O/$a/$h" --um 9.596 --period $st --n 24 --min 8 --threads 8 2>&1 | tail -3 | tr '\n' ' ')"; done
    [ $a = B ] || for h in rv ink c0; do [ -e "$O/B/$h/zarr.json" ] && echo "  diff vs B $h: $($U z3-diff "$O/$a/$h" 9.596 "$O/B/$h" 9.596 --threads 8 2>&1 | tail -2 | tr '\n' ' ')"; done
  done
} > "$O/seam.txt"
cat "$O/seam.txt"

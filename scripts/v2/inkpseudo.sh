#!/usr/bin/env bash
# scripts/v2/inkpseudo.sh [S ...]: v2 pseudo ink targets = the scroll's ift3-24000 ink prediction restricted to the recto band of its
# snapped-wrap labels (V2LABEL: V2/labels/<S>/phase.zarr/<KEY>, phase < 126 = k .. k+0.5; ignored voxels already 255) -> prob254
# V2/ink/pseudo/<S>.zarr/<KEY> (ufsm ink-gate: p x 254 in the band, 255 elsewhere) + <S>.txt (gated voxels, p >= .5 share, mm3).
# Ink source + model + seam handling per scroll: scripts/v2/inksrc.sh (single-pass inks get --seam: window borders unknown).
# Default scrolls: the train rows of V2/scrolls.tsv with labels and no <S>.done. Env: V2, U (V2/src/build/ufsm), TH (16), LO/HI (0/125).
set -u
H=$(cd "$(dirname "$0")" && pwd); V2=${V2:-/vesuvius/ufsm/v2}; U=${U:-$V2/src/build/ufsm}; O=$V2/ink/pseudo; mkdir -p $O
list=("$@"); [ ${#list[@]} -gt 0 ] || list=($(awk -F'\t' 'NR > 1 && $2 == "train" { print $1 }' $V2/scrolls.tsv))
for S in "${list[@]}"; do
  [ -e $O/$S.done ] && continue
  L=$V2/labels/$S/phase.zarr; [ -d $L ] || { echo "inkpseudo $S: no labels yet ($L)"; continue; }
  key=$(ls $L | grep -v '\.json$' | head -1); [ -n "$key" ] || { echo "inkpseudo $S: empty $L"; continue; }
  IFS=$'\t' read -r src ik model mode seam < <(bash $H/inksrc.sh $S); [ "$src" != - ] || { echo "inkpseudo $S: no ink zarr"; continue; }
  sm=; [ "$seam" != - ] && sm="--seam $seam"
  rm -rf $O/$S.zarr
  if nice -n 10 $U ink-gate $O/$S.zarr --ink $src $ik --ink-um $ik --phase $L $key --phase-um $key --lo ${LO:-0} --hi ${HI:-125} $sm \
       --threads ${TH:-16} > $O/$S.txt 2>&1; then
    echo "$src $ik $model $mode $seam $(date '+%F %T')" > $O/$S.done; echo "inkpseudo $S: $(tail -n 1 $O/$S.txt)"
  else echo "inkpseudo $S: FAILED ($O/$S.txt)"; fi
done

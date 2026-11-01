#!/usr/bin/env bash
# orcfit.sh FIELDS LABELS TAG Z NZ [Y X NY NX] : label-oracle slab fit (0.5 "fit-validate every label set").
#   FIELDS: the scroll's fit-field dir (19.2 / 18.7 / 17.3 um cells, native = the labels' grid); LABELS: surf-labels output dir
#   ({codes,phase}.zarr/<key>); the slab = field cells [Z, Z+NZ) x [Y, Y+NY) x [X, X+NX) (default the whole section).
#   fields-oracle --fill net (label phase where known, the network's elsewhere) -> fit (GPU under $LOCK) scored against the labels'
#   own codes (and against GT=ROOT,KEY when given). Output $O/TAG/{fit.log,gt.log}, one summary line on stdout.
# Env: U, O (/mnt/scratch/v2/val/orc), GPU (0), LOCK (/vesuvius/ufsm/opt/gpu$GPU.lock), GT, FITX (extra fit flags).
set -uo pipefail
F=$1 L=$2 T=$3 Z=$4 NZ=$5
U=${U:-/mnt/scratch/v2/v2label/src/build/ufsm}; O=${O:-/mnt/scratch/v2/val/orc}; GPU=${GPU:-0}; LOCK=${LOCK:-/vesuvius/ufsm/opt/gpu$GPU.lock}
H=$(dirname "$(readlink -f "$0")")/../wbench
read -r sz sy sx <<< "$(jq -r '.shape | map(tostring) | join(" ")' $F/fields.json)"
Y=${6:-0} X=${7:-0} NY=${8:-$sy} NX=${9:-$sx}
d=$O/$T; rm -rf $d; mkdir -p $d
key=$(ls $L/phase.zarr | grep -v zarr.json | head -1)
bash $H/mkcrop.sh $F $d/f $Z $Y $X $NZ $NY $NX > $d/crop.log || exit 1
$U fields-oracle $d/f $L/phase.zarr $key $d/o --fill net > $d/orc.log 2>&1 || { echo "$T fields-oracle failed"; exit 1; }
rm -rf $d/f
flock $LOCK env UFSM_FIT_THREADS=${FTH:-12} nice -n 5 $U fit $d/o --gpu 0 --out $d/fit --codes $L/codes.zarr $key --score-pairs 1 --dir 1 --uw-repair 8 ${FITX:-} > $d/fit.log 2>&1 \
  || { echo "$T fit failed"; exit 1; }
g=""
if [ -n "${GT:-}" ]; then
  flock $LOCK nice -n 5 $U fit $d/o --gpu 0 --model $d/fit --codes ${GT%,*} ${GT#*,} --score-pairs 1 > $d/gt.log 2>&1
  g="gt ids $(grep -oE 'winding ids right [0-9.]+' $d/gt.log | tail -1 | awk '{print $4}') gt pairs $(grep -oE '0.5 mm [0-9.]+|4 mm [0-9.]+' $d/gt.log | head -2 | awk '{print $3}' | tr '\n' ' ')"
fi
echo "$T label share $(grep -oE 'share [0-9.]+' $d/orc.log | head -1 | awk '{print $2}') ids $(grep -oE 'winding ids right [0-9.]+' $d/fit.log | tail -1 | awk '{print $4}') pairs $(grep -oE '0.5 mm [0-9.]+|4 mm [0-9.]+' $d/fit.log | head -2 | awk '{print $3}' | tr '\n' ' ') $g"
rm -rf $d/o

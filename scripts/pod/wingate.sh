#!/usr/bin/env bash
# scripts/pod/wingate.sh: the gate for larger wind96 fields windows (predict --window auto, opt-in, UNGATED until this passes).
# NOT RUN YET (10-31: no pod rented for it). Needs a GPU with >= ~32 GB for the larger windows (direct slabs: ~73 B per window voxel;
# 640 x 1152^2 = 850 M voxels -> UFSM_SLAB_DIRECT=0 (~46 B, ~39 GB) is set for it below on a 48 GB card).
#
# Per scroll crop (Paris 4 + one crushed scroll) and window variant:
#   fields  predict --fit-zarr over the crop box with that window (the production flags of scripts/unroll/unroll.sh otherwise)
#   ref     count-field --model-only (stride 1)      slabs  NS snapped slab fits of 64 cells (+16 overlap)
#   join    fit-join --ref                            wrap-check against the CT;  cf-compare / fit-compare vs the base variant
#   P4 GT   fit --codes (published winding codes): ids / pairs of the joined model and of the ref
# Variants (z x xy, halo 64; the strides --fit-zarr accepts with --shard 512: z 256 / 512, xy 512 / 1024):
#   base 384 x 640^2 (production, 43% of the computed voxels interior)   c640 640^3 (51%)   s1152 384 x 1152^2 (50%)   w1152 640 x 1152^2 (63%)
# (768^3 / 896^3 at halo 64 have strides 640 / 768: not a divisor or a multiple of the 512 shard, rejected by --fit-zarr.)
# PASS (provisional, the fieldq / window-overlap noise floor): per scroll, every variant's join agreement and wrap-check per fitted turn
# within DAGREE (0.01) / DWC (0.01) of base, Paris 4 GT ids of the joined model >= base - DGT (0.002). Prints PASS / FAIL per variant.
# Output: $G/<scroll>/<variant>/..., $G/results.txt. Usage: scripts/pod/wingate.sh [variants...]  (default: base c640 s1152 w1152)
# Env (pod paths; desk defaults): U, CK, MKCROP, G (out dir), GPU, TH,
#   P4R P4K P4UM P4AX P4BOX P4GT   (CT root, key at level L, um at L, axis json, box at L, GT codes zarr)
#   CRS CRR CRK CRUM CRAX CRBOX    (crushed scroll: name, CT root, key, um, axis json, box)
set -u
H=$(cd "$(dirname "$0")" && pwd)
U=${U:-$H/../../build/ufsm}; MKCROP=${MKCROP:-$H/../wbench/mkcrop.sh}
CK=${CK:-/vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt}
G=${G:-$PWD/wingate}; GPU=${GPU:-0}; TH=${TH:-8}; NS=${NS:-4}; I=64; OV=16
DAGREE=${DAGREE:-0.01}; DWC=${DWC:-0.01}; DGT=${DGT:-0.002}
P4R=${P4R:-/vesuvius/ufsm/volcomp}; P4K=${P4K:-PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr}; P4L=${P4L:-2}; P4UM=${P4UM:-2.4}
P4AX=${P4AX:-/vesuvius/ufsm/paris4/ckpt/axis-0.json}; P4BOX=${P4BOX:-9216,2048,1536,512,5120,5120}; P4GT=${P4GT:-/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr}
CRS=${CRS:-PHerc1299}; CRR=${CRR:-/vesuvius/ufsm/volcomp/PHerc1299/volumes}; CRK=${CRK:-20260309130042-2.399um-0.2m-78keV-masked.zarr}; CRL=${CRL:-2}; CRUM=${CRUM:-2.399}
CRAX=${CRAX:-}; CRBOX=${CRBOX:-}   # (axis: `ufsm axis CRR CRK axis.json` once; box: a 512-deep, 5120^2 crop around the axis at mid z, at level CRL)
VARS=${*:-base c640 s1152 w1152}
mkdir -p $G
win() { case $1 in base) echo "384 640";; c640) echo "640 640";; s1152) echo "384 1152";; w1152) echo "640 1152";; *) return 1;; esac; }
num() { grep -oE "$1" "$2" 2> /dev/null | grep -oE '[0-9.]+$' | head -1; }

run1() {   # run1 SCROLL VARIANT R K L UM AX BOX [GT]
  local S=$1 V=$2 R=$3 K=$4 L=$5 UM=$6 AX=$7 BOX=$8 GT=${9:-} wz w O args=() k c0 c1 a b d dir ze ny nx t0
  read wz w < <(win $V) || { echo "unknown variant $V"; return 1; }
  O=$G/$S/$V; rm -rf $O; mkdir -p $O; t0=$(date +%s)
  local ENVX=(); [ $w -ge 1152 ] && ENVX=(UFSM_SLAB_DIRECT=0)
  env "${ENVX[@]}" $U predict $CK $R $K $O/f.tmp --um $UM --level $L --box $BOX --window-z $wz --halo-z 64 --window $w --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads $TH --gpu $GPU --axis $AX --levels 1 --fit-zarr $O/f --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 > $O/fields.log 2>&1 \
    || { echo "$S $V FAIL fields"; return 1; }
  rm -rf $O/f.tmp; local tf=$(( $(date +%s) - t0 ))
  $U count-field $O/f $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $O/ref --model-only 1 --model-stride 1 --threads $TH --mem-gb 16 > $O/ref.log 2>&1 || { echo "$S $V FAIL ref"; return 1; }
  rm -rf $O/cf.unused
  dir=$(jq -r .dir $O/ref/model.json); ze=$(jq -r '.shape[0]' $O/f/fields.json); ny=$(jq -r '.shape[1]' $O/f/fields.json); nx=$(jq -r '.shape[2]' $O/f/fields.json)
  local ns=$(( ze / I )); [ $ns -gt $NS ] && ns=$NS
  for k in $(seq 0 $((ns - 1))); do
    c0=$((k * I)); c1=$((c0 + I)); a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
    d=$O/s$k; bash $MKCROP $O/f $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null || { echo "$S $V FAIL crop $k"; return 1; }
    env UFSM_FIT_THREADS=$TH $U fit $d/f --gpu $GPU --out $d/fit --dir $dir --ref-model $O/ref --ref-snap 1 --uw-repair 0 > $d/fit.log 2>&1 || { echo "$S $V FAIL fit $k"; return 1; }
    args+=($d/fit $((2 * c0)) $((2 * c1))); rm -rf $d/f
  done
  $U fit-join $O/joined $O/f --blend 16 --ref $O/ref "${args[@]}" > $O/join.log 2>&1 || { echo "$S $V FAIL join"; return 1; }
  $U wrap-check $O/joined $O/f --ct $R $K/$L --ct-um $UM --tsv $O/wc.tsv --threads $TH > $O/wc.log 2>&1 || { echo "$S $V FAIL wrap-check"; return 1; }
  if [ $V != base ] && [ -e $G/$S/base/ref/model.json ]; then
    $U cf-compare $G/$S/base/ref $O/ref $O/f --threads $TH > $O/cfc.log 2>&1; $U fit-compare $G/$S/base/f $O/f > $O/fc.log 2>&1
  fi
  if [ -n "$GT" ]; then
    $U fit $O/f --model $O/joined --codes $GT 9.6 --score-pairs 1 > $O/gt.log 2>&1; $U fit $O/f --model $O/ref --codes $GT 9.6 --score-pairs 1 > $O/gtref.log 2>&1
  fi
  local ag=$(num 'integer agreement[^:]*: [0-9.]+' $O/join.log) wc=$(num 'per fitted turn [0-9.]+' $O/wc.log) gt=$(num 'ids[^0-9]*[0-9.]+' $O/gt.log)
  echo "$S $V fields ${tf}s | join agree ${ag:--} | wc ${wc:--} | gt ids ${gt:--} | $(grep -oE '[0-9.]+ s/window|tiles in [0-9]+s' $O/fields.log | tail -1) | total $(( $(date +%s) - t0 ))s" | tee -a $G/results.txt
  echo "$S $V ${ag:-nan} ${wc:-nan} ${gt:-nan}" >> $G/metrics.txt
}

: > $G/metrics.txt
for V in $VARS; do
  run1 P4 $V $P4R $P4K $P4L $P4UM $P4AX $P4BOX $P4GT
  if [ -n "$CRAX" ] && [ -n "$CRBOX" ]; then run1 $CRS $V $CRR $CRK $CRL $CRUM $CRAX $CRBOX; else echo "$CRS: set CRAX and CRBOX (crushed-scroll crop) to gate it too"; fi
done
# PASS / FAIL vs base per scroll
awk -v da=$DAGREE -v dw=$DWC -v dg=$DGT '
  { k = $1; v = $2; ag[k, v] = $3; wc[k, v] = $4; gt[k, v] = $5; if ($2 == "base") { b[k] = 1 } else vs[k, v] = 1; sc[k] = 1 }
  END { for (kv in vs) { split(kv, p, SUBSEP); k = p[1]; v = p[2]; if (!b[k]) { print k, v, "NO BASE"; continue }
          ok = (ag[k, v] + da >= ag[k, "base"]) && (wc[k, v] - wc[k, "base"] <= dw && wc[k, "base"] - wc[k, v] <= dw) && (gt[k, v] == "nan" || gt[k, v] + dg >= gt[k, "base"])
          printf "%s %s %s (agree %s vs %s, wc %s vs %s, gt %s vs %s)\n", k, v, ok ? "PASS" : "FAIL", ag[k, v], ag[k, "base"], wc[k, v], wc[k, "base"], gt[k, v], gt[k, "base"] } }' $G/metrics.txt | tee -a $G/results.txt

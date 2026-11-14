#!/usr/bin/env bash
# scripts/rv/eval.sh CKPT TAG [SCROLLS "p1667 p139 p4"]: counting eval of a recto / verso checkpoint (train --rv) with RECTO's
# recto-ruler (turns per GT turn / share off >= 0.5 turn by radius and by spacing), on the same rays / z levels as RECTO's m7 baselines:
#   p1667 (eval scroll, never trained on): pn2 ref + fields /vesuvius/ufsm/v2/count/p1667, GT codes 4.798, --zr 500,8800 --nz 24 (m7 .973/.099)
#   p139  (trained outside its 8 wbench boxes): count/p139, GT codes 18.72, --zr 1000,12000 --nz 24 (m7 .847/.220)
#   p4    (trained outside PH1-3/HOLD: CONTAMINATED for this model): count/p4w + cfit/whole/refcrop, q-9p6, t1.sh zs + --zblock 1500,1 (m7 .920/.210)
# Predictions: predict --channel rv (1 + 254 (1 + p_recto - p_verso) / 2, 0 = CT 0) only on the shard rows (z) the ruler samples (--full-grid 1, --box per row,
# --shard SH 128, --occupancy 5), on the scan level the fields use; then recto-ruler with --src rv (peaks of the fused signal = recto-side
# runs: one per sheet even where sheets touch) and --src rvr (the same volume, hit = run, --thr 160).
# Out: $E/<TAG>/<scroll>/{pred,ruler.log}; summary $E/<TAG>.txt. Prediction zarrs are deleted after the ruler unless KEEP=1.
# WZ="" for latent-grid checkpoints (train --lat 4: cubic windows only), WINDOW (640).
# Env: U (rv binary), RU (ruler binary: RECTO's src2), GPU (1), LOCK, E (/vesuvius/ufsm/v2/rv/eval), TH (16), WIN (640/64, z 192/64).
# LATENTS=STORE: predict from that ufsm lc-encode store of the scroll's CT level instead of the CT (--latents; one scroll per run, e.g.
# SC=p1667 with /mnt/scratch/lat/p1667-l2.lc; --lat-basis checkpoints); each row's predict log then ends with its read accounting (UFSM_PRED_IO).
set -u
CK=$1; T=$2; SC=${3:-p1667 p139 p4}
U=${U:-/vesuvius/ufsm/v2/rv/bin/ufsm.ev}; RU=${RU:-/vesuvius/ufsm/v2/recto/src2/build/ufsm}; GPU=${GPU:-1}; LOCK=${LOCK-/vesuvius/ufsm/opt/gpu$GPU.lock}
E=${E:-/mnt/scratch/rv/eval}; TH=${TH:-16}; SH=${SH:-128}; CTR=${CTR:-/vesuvius/ufsm/v2/ct}; C=${CNT:-/vesuvius/ufsm/v2/count}
# off the desk (2026-11-11, scripts/univ): CTR=<volcomp mirror> VOLP=volumes/ (the mirror keeps scans under <scroll>/volumes/), CNT, RVGT (GT root), RU
RVGT=${RVGT:-/vesuvius/ufsm/gt}; VOLP=${VOLP:-}
mkdir -p $E/$T; echo "== $T $CK $(date)" >> $E/$T.txt
zlist() {   # zlist ZR0 ZR1 NZ: the ruler's --nz z levels (model-local native z)
  awk -v a=$1 -v b=$2 -v n=$3 'BEGIN { for (i = 0; i < n; i++) printf "%s%d", i ? "," : "", int(a + (i + 0.5) * (b - a) / n) }'; }
for S in $SC; do
  case $S in
    p1667) M=$C/p1667/pn2/ref; F=$C/p1667/fields; Q="$RVGT/PHerc1667-wind-20261016/codes 4.798"; CK_=PHerc1667/${VOLP}20251217075048-2.399um-0.2m-78keV-masked.zarr; LV=2; UM=9.596; U0=2.399
           ZS=$(zlist 500 8800 24); ZB="";;
    p139)  M=$C/p139/pn2/ref; F=$C/p139/fields; Q="$RVGT/PHerc0139-wind-20261016/codes 18.72"; CK_=PHerc0139/${VOLP}20250728140407-9.362um-1.2m-113keV-masked.zarr; LV=0; UM=9.362; U0=9.362
           ZS=$(zlist 1000 12000 24); ZB="";;
    p4)    M=$C/p4w/pn2/ref; F=/vesuvius/ufsm/cfit/whole/refcrop; Q="$RVGT/paris4-q9p6/q-9p6.zarr 9.6"; CK_=PHercParis4/${VOLP}20260411134726-2.400um-0.2m-78keV-masked.zarr; LV=2; UM=9.6; U0=2.4
           ZS=8008,8040,8072,8104,9508,9540,9572,9604,11008,11040,11072,11104,12508,12540,12572,12604,14008,14040,14072,14104,15508,15540,15572,15604; ZB="--zblock 1500,1";;
    *) echo "unknown $S"; continue;;
  esac
  D=$E/$T/$S; mkdir -p $D
  AXF=$(bash $(dirname $(readlink -f $0))/../unroll/axis_for.sh $S) || { echo "$S: no axis (predict needs one)" | tee -a $E/$T.txt; continue; }
  read -r oz oy ox mz my mx h < <(jq -r '"\(.origin[0]) \(.origin[1]) \(.origin[2]) \(.shape[0]) \(.shape[1]) \(.shape[2]) \(.h)"' $M/model.json)
  # shard rows to predict: those holding a sampled z (global native z = local + model origin); --zblock: only the kept blocks
  rows=$(echo $ZS | tr , '\n' | awk -v o=$oz -v sh=$SH -v zb="$ZB" '{ z = $1 + 0.5 + o; if (zb != "") { split(zb, a, "[ ,]"); b = a[2]; p = a[3]; if (int(z / b) % 2 != p) next }
         print int(z / sh) }' | sort -un)
  if [ ! -e $D/pred.done ]; then
    t0=$(date +%s)
    LX=(); [ -n "${LATENTS:-}" ] && LX=(--latents "$LATENTS")
    for r in $rows; do
      L=(); [ -n "$LOCK" ] && L=(flock $LOCK)
      [ -n "${K:-}" ] && [ -d $K/gpu.shared ] && while [ -e $K/gpu.pause ] || [ $((10#$(date +%M))) -ge $([ -e $K/gpu.gate-waiting ] && echo 35 || echo 15) ]; do sleep 5; done   # shared pod GPU ($K/GPU.md): wait out exclusive timing slots
      CUDA_VISIBLE_DEVICES=$GPU UFSM_PRED_IO=1 "${L[@]}" nice -n 5 $U predict $CK $CTR $CK_ $D/pred --um $U0 --level $LV --box $((r * SH)),$oy,$ox,$SH,$((my * h)),$((mx * h)) \
        --full-grid 1 --resume 1 $AXF --shard $SH --window ${WINDOW:-640} --halo 64 ${WZ---window-z $((SH + 64)) --halo-z 32} --levels 1 --channel rv \
        --threads $TH --gpu 0 --q 0 ${PX:-} "${LX[@]}" >> $D/pred.log 2>&1 || { echo "$S: predict row $r failed (see $D/pred.log)" | tee -a $E/$T.txt; continue 2; }
    done
    echo "$S: predicted $(echo $rows | wc -w) shard rows of $SH in $(( $(date +%s) - t0 )) s" | tee -a $E/$T.txt; touch $D/pred.done
  fi
  [ "${PREDONLY:-0}" = 1 ] && continue   # the GPU phase of a two-phase eval: the ruler follows in the second run
  key=$(ls $D/pred | grep -v json | head -1)
  nice $RU recto-ruler $M $F --codes $Q --src rv $D/pred $key $UM --zs $ZS $ZB --threads $TH > $D/ruler.log 2> $D/ruler.err
  echo "$S $(grep -E '^ +all' $D/ruler.log | tail -1)" | tee -a $E/$T.txt
  [ "${KEEP:-0}" = 1 ] || rm -rf $D/pred
done

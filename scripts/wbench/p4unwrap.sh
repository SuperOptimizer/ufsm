#!/usr/bin/env bash
# p4unwrap.sh CKPT TAG: the whole-Paris-4 unwrap milestone (the 10-21 desk run_pred + run_fit + post_slabs, parameterized).
#   1. fit fields of the whole scroll (9.6 um input, 19.2 um cells, s384x640 windows, no TTA)        -> $O/fields
#   2. full cross-section z slabs: 64-cell interiors + 16 overlap on the union crop, fit --uw-repair 8 -> $O/sNNN (one lane per GPU)
#   3. fit-join of all slabs, whole-scroll --codes score (radius detail), slip map                     -> $O/joined-score.log, $O/summary.txt
#   4. WRAPS=1: wraps (.sfc step 4) + wrapswitch per wrap.
# Baseline: wind96 10-21 whole-scroll joined ids 0.1117 (desk /vesuvius/ufsm/paris4/slabs-base/joined-score.log).
# Cost (wind96 10-21, one 5060 Ti): predict 1.4 h (host 10.6 GB, GPU 7.6 GB, 50 GB fields), 147 slab fits 4.7 GPU-h (~21 GB host
#   per fit), join + score ~1.6 h (mostly CPU). Restart-safe (state files, per-slab done markers); stop with touch $O/STOP.
# Env: SITE (desk|laptop, the wbench site env), GPUS ("0" | "0 1": fit lanes, slab k -> lane k mod n; predict on the first),
#   U (binary, default $WB/bin/ufsm), O ($WB/p4unwrap/TAG), FIELDS (existing fit-field dir: skip the predict),
#   Z0 ZE Y0 NY X0 NX (crop in field cells; default the whole scroll), I OV (64 / 16), ONLY ("k1 k2": just these slabs),
#   FITARGS (extra fit flags), WRAPS (0), NOLOCK=1 (ignore the site GPU lock; the laptop lock is taken per GPU step).
set -u
H=$(cd "$(dirname "$0")" && pwd)
CK=$(readlink -f "$1"); TAG=$2
SITE=${SITE:-$([ -d $HOME/ufsm-data ] && echo laptop || echo desk)}; GPUS=${GPUS:-0}; GPU=${GPUS%% *}
. $H/site-$SITE.env
U=${U:-$WB/bin/ufsm}; [ -x $U ] || { echo "no binary $U"; exit 2; }
O=${O:-$WB/p4unwrap/$TAG}; mkdir -p $O
V=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; Q=$GT/paris4-q9p6/q-9p6.zarr
Z0=${Z0:-128}; ZE=${ZE:-9473}; Y0=${Y0:-608}; NY=${NY:-3328}; X0=${X0:-352}; NX=${NX:-3520}; I=${I:-64}; OV=${OV:-16}
F=${FIELDS:-$O/fields}; BASE=0.1117
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $O/run.log; }
stopped() { [ -e $O/STOP ]; }
space() { local f=$(df -BG --output=avail $O | tail -n 1 | tr -dc 0-9); [ $f -ge 120 ] || { log "only $f GB free"; return 1; }; }
LK=; [ -z "${NOLOCK:-}" ] && [ "$SITE" = laptop ] && LK=${LOCK:-}
GQ=$H/gpuq.sh; [ -e $GQ ] || GQ=$H/../gpuq.sh; GPUQ_LOCK=${LK:-$HOME/ufsm-data/gpu.lock}; . $GQ
G() { if [ -n "$LK" ]; then ( gpuq_acquire ${PRIO:-bg} p4unwrap-$TAG; "$@"; rc=$?; gpuq_release; exit $rc ); else "$@"; fi; }   # a GPU step through the laptop gpuq (per step, not per run)
log "p4unwrap $TAG ckpt $CK bin $(sha1sum $U | cut -c1-8) gpus $GPUS fields $F crop z $Z0..$ZE y $Y0+$NY x $X0+$NX"
echo "$CK" > $O/ckpt

# ---- 1. fields ----
if [ -z "${FIELDS:-}" ] && [ ! -e $O/fields.done ]; then
  space || exit 3; rm -rf $F $F.tmp; t0=$(date +%s); log "predict start"
  G RUNBIG $U predict $CK $P4ROOT $V $F.tmp --um 2.4 --level 2 --window-z 384 --halo-z 64 --window 640 --halo 64 \
      --shard 512 --grid-origin 0,0,0 --ema 1 --q 8 --threads $THREADS --gpu $GPU --axis $AXP4 --levels 1 --fit-zarr $F --fit-h 2 --fit-q 8 \
      --fit-q-phase 0 --no-out 1 > $O/pred.log 2>&1 || { log "predict FAILED (pred.log)"; exit 1; }
  rm -rf $F.tmp; echo "$(( $(date +%s) - t0 ))" > $O/fields.done; log "predict done $(( $(date +%s) - t0 )) s $(du -sh $F | cut -f1)"
fi
[ -e $F/fields.json ] || { log "no fields $F"; exit 1; }

# ---- 2. slab fits ----
NS=$(( (ZE - Z0 + I - 1) / I ))
order() {   # the codes region (cells 3456..8576) first, middle outward, then the rest bottom-up (as run_fit.sh)
  local k c; for k in $(seq 0 $((NS - 1))); do c=$((Z0 + k * I)); [ $c -ge 3456 ] && [ $c -lt 8576 ] && echo "$(( (c - 6016) * (c - 6016) )) $k"; done | sort -n | awk '{print $2}'
  for k in $(seq 0 $((NS - 1))); do c=$((Z0 + k * I)); { [ $c -lt 3456 ] || [ $c -ge 8576 ]; } && echo $k; done; }
LIST=${ONLY:-$(order)}
lane() {   # lane L GPU: the slabs k of LIST with k mod NL == L
  local L=$1 g=$2 k s d c0 c1 a b t0 rc ids e08
  for k in $LIST; do
    [ $((k % NL)) = $L ] || continue; stopped && { log "STOP"; return 0; }; space || return 3
    s=$(printf 's%03d' $k); d=$O/$s; [ -e $d/done ] && continue
    c0=$((Z0 + k * I)); c1=$((c0 + I)); [ $c1 -gt $ZE ] && c1=$ZE
    a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ZE ] && b=$ZE
    rm -rf $d; mkdir -p $d
    bash $H/mkcrop.sh $F $d/f $a $Y0 $X0 $((b - a)) $NY $NX > /dev/null || { log "$s mkcrop failed"; continue; }
    echo "$c0 $c1 $a $b" > $d/cells; t0=$(date +%s)
    G RUNBIG env UFSM_FIT_THREADS=$THREADS $U fit $d/f --gpu $g --out $d/fit --codes $Q 9.6 --score-detail 1 --dir 1 --uw-repair 8 ${FITARGS:-} > $d/fit.log 2>&1
    rc=$?
    ids=$(grep -oE 'winding ids right [0-9.]+' $d/fit.log | tail -1 | awk '{print $4}'); e08=$(grep -oE '< 0.08 turn [0-9.]+' $d/fit.log | tail -1 | awk '{print $4}')
    echo "$s cells $c0..$c1 rc $rc ids ${ids:-na} e08 ${e08:-na} gpu $g $(( $(date +%s) - t0 ))s" >> $O/slabs.txt
    [ $rc = 0 ] && touch $d/done || log "$s fit rc $rc"
  done; }
NL=$(echo $GPUS | wc -w); L=0; for g in $GPUS; do lane $L $g & L=$((L + 1)); done; wait
stopped && exit 0
nd=$(ls -d $O/s???/done 2>/dev/null | wc -l); log "slab fits: $nd of $NS done ($(awk '{s += $NF + 0} END {printf "%.2f", s / 3600}' $O/slabs.txt) GPU-h)"

# ---- 3. join + whole-scroll score ----
FC=$O/fields-crop; bash $H/mkcrop.sh $F $FC $Z0 $Y0 $X0 $((ZE - Z0)) $NY $NX > /dev/null
for d in $O/s???; do [ -e $d/done ] || continue; read -r c0 c1 a b < $d/cells   # z x radius map: slab, cells, radius bin, voxels, error fractions -2..+2
  grep -E '^    radius' $d/fit.log | awk -v s=$(basename $d) -v c0=$c0 -v c1=$c1 '{print s"\t"c0"\t"c1"\t"$2"\t"$4"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}'
done > $O/slipmap.tsv
args=(); for d in $(ls -d $O/s??? | sort); do [ -e $d/done ] || continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * (c0 - Z0))) $((2 * (c1 - Z0)))); done
RUN $U fit-join $O/joined $FC --blend 16 "${args[@]}" > $O/join.log 2>&1 || { log "fit-join failed"; tail -n 3 $O/join.log; exit 1; }
G RUNBIG env UFSM_FIT_THREADS=$THREADS $U fit $FC --model $O/joined --gpu $GPU --codes $Q 9.6 --score-detail 1 --score-pairs 1 > $O/joined-score.log 2>&1
ids=$(grep -oE 'winding ids right [0-9.]+' $O/joined-score.log | tail -1 | awk '{print $4}')
js=$(grep -oE "mean share\(mode\).*: [0-9.]+" $O/join.log | awk "{print \$NF}"); js=${js:-na}
{ echo "tag $TAG ckpt $CK"; echo "whole-scroll joined ids ${ids:-na} (wind96 10-21: $BASE), mean join share $js, slabs $nd of $NS"
  grep -E 'pair winding right' $O/joined-score.log | tail -n 1
  grep -E '^    radius' $O/joined-score.log; } > $O/summary.txt
log "joined ids ${ids:-na} vs wind96 $BASE, join share $js"

# ---- 4. wraps ----
if [ "${WRAPS:-0}" = 1 ]; then
  rm -rf $O/wraps; RUN $U wraps $O/joined $O/wraps --step 4 --out-um 9.6 --sfc 0.25 --fields $FC --threads $THREADS > $O/wraps.log 2>&1
  RUN $U wrapswitch $O/wraps $Q 9.6 --axis $AXP4 --dir 1 --range 30 --threads $THREADS --per-wrap > $O/wraps.switch.log 2>&1
fi
log "p4unwrap $TAG finished"

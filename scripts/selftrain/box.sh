#!/usr/bin/env bash
# box.sh STAGE SCROLL I: one pseudo-label box of a self-training round (workstream B, 2026-10-25). Box I of A/boxes/<SCROLL>.json
# (level-L voxels of the scan in scans.tsv, 1024^3, origin on the 512 shard grid).
#   STAGE pred : the round's teacher CK -> fit fields A/r$R/f/<SCROLL>-b<I> (predict --fit-zarr h 2 + --fit-ct planes), GPU
#   STAGE fit  : global / local fits of the box, or of its 8 octants where the box labels little (best by agree-gated share) -> A/r$R/fit/<SCROLL>-b<I>[.o<k>] + .units
#   STAGE label: gated stored-phase codes (fit --winding-mod14 3, PHASE_* 0..251) written straight into the scan's global label zarrs
#                A/r$R/labels/<ARM>/<SCROLL> (512^3 shards, one per box octant), one per arm in ARMS (NAME=CONF,ERR,MAT);
#                the per-arm gate stats -> A/r$R/stats/<SCROLL>-b<I>.<ARM>.txt; KEEPF=0 deletes the fields afterwards.
# The scan's training level: 0 (8.64 / 9.36 um scans) or 2 (2.4 um scans, 9.6 um); predict, fit and labels all on that level's grid (step 1).
# Env: A (/vesuvius/ufsm/selftrain), R (round, 0), CK (teacher), U (binary), GPU (0), CPU=1 (fit on the CPU), TH (8), ARMS, PSIGN (1)
set -u
STAGE=$1 S=$2 I=$3; A=${A:-/vesuvius/ufsm/selftrain}; R=${R:-0}; U=${U:-$A/src/build/ufsm}; G=${GPU:-0}; TH=${TH:-8}
D=$(dirname "$(readlink -f "$0")"); RD=$A/r$R; N=$S-b$I
read -r _ SC LV UM VOL AX NB ST <<< "$(awk -F'\t' -v s=$S '$1 == s' $D/scans.tsv)"
[ -n "${VOL:-}" ] || { echo "$N: not in scans.tsv"; exit 2; }
BOX=$(jq -r ".[$I] | map(tostring) | join(\",\")" $A/boxes/$S.json); [ "$BOX" != null ] || { echo "$N: no box $I"; exit 2; }
CT=/vesuvius/ufsm/u0/vol; F=$RD/f/$N; M=$RD/fit/$N; mkdir -p $RD/f $RD/fit $RD/stats $RD/logs; L=$RD/logs/$N
STEP=1   # (the fields' native grid is the training level itself: labels on the training-level grid)
case $STAGE in
pred)
  [ -s $F/fields.json ] && { echo "$N: fields exist"; exit 0; }
  rm -rf $F $F.scratch
  nice -n 5 $U predict ${CK:?teacher CK} $CT $VOL $F.scratch --um $UM --level $LV --box $BOX --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads $TH --gpu $G --axis $AX --levels 1 --fit-zarr $F --fit-h 2 --fit-ct 1 --no-out 1 > $L.pred.log 2>&1 \
    || { echo "$N: predict failed: $(tail -1 $L.pred.log)"; rm -rf $F.scratch; exit 1; }
  rm -rf $F.scratch; echo "$N: fields $(du -sh $F | cut -f1)";;
fit)
  # units = (fields, model) pairs labelled by STAGE label, listed in $M.units. Candidates per unit: the global spiral fit (g) and two
  # local fits (p / m: axis 2000 voxels from the unit centre along -/+ its mean sheet normal, fit --axis-normal, --core 0: crushed
  # scans), scored by the agree-gated share (AGREE, step-4 throwaway raster); the best one is kept. The whole box: g only; each octant: g, + p / m when
  # g keeps under LOCALMIN (0.6) of the confident material = the gate without its error test); where the whole box keeps under that,
  # also its 8 octants (512^3, fields views $F.o<k>: "zarr_offset" + the axis table sliced) and the octants win when their mean share is
  # larger (OCT=0: never). Scores in $M.choice.
  [ -e $M.done ] && { echo "$N: fit exists"; exit 0; }
  [ -s $F/fields.json ] || { echo "$N: no fields"; exit 1; }
  DEV=(--gpu $G); [ "${CPU:-0}" = 1 ] && DEV=(--cpu --threads $TH)
  AG=${AGREE:-0.3,0.15,0.3}; LM=${LOCALMIN:-0.6}; : > $M.choice
  share() {   # fields model gate -> "share% cos" (step-4 throwaway raster)
    local t=$2.score.tmp; rm -rf $t
    nice -n 5 env UFSM_FIT_THREADS=$TH $U fit $1 "${DEV[@]}" --model $2 --winding $t --winding-mod14 3 --winding-step 4 --winding-gate $3 > $t.log 2>&1 || { rm -rf $t $t.log; echo "0 0"; return; }
    sed -nE 's/.*labelled \(([0-9.]+)%\); conf-weighted mean cos 2 pi \(W - phase\) ([-0-9.]+).*/\1 \2/p' $t.log | tail -1; rm -rf $t $t.log; }
  fitone() {  # fields model-prefix name flags.. -> "name share cos loss" (or "name failed")
    local f=$1 mp=$2 k=$3; shift 3; rm -rf $mp.$k
    nice -n 5 env UFSM_FIT_THREADS=$TH $U fit $f "${DEV[@]}" --out $mp.$k --dir 0 --uw-repair 8 --w-ct 1 --ct-phi 0 --ct-flood 0 "$@" > $mp.$k.log 2>&1 \
      || { echo "$k failed"; return; }
    echo "$k $(share $f $mp.$k $AG) $(grep -oE 'loss [0-9.]+' $mp.$k.log | tail -1)"; }
  unit() {    # fields model [nolocal] -> fit candidates, keep the best as <model>; prints "best share confident"
    local f=$1 mp=$2 c=$2.cand; : > $c
    fitone $f $mp g >> $c
    local cf=$(share $f $mp.g ${AG%%,*},9,${AG##*,} | cut -d' ' -f1); local sg=$(awk '$1 == "g" {print $2 + 0}' $c)
    if [ -z "${3:-}" ] && awk -v s=${sg:-0} -v m=${cf:-0} -v f=$LM 'BEGIN { exit !(s < f * m) }'; then
      fitone $f $mp p --core 0 --axis-normal 2000 >> $c; fitone $f $mp m --core 0 --axis-normal -2000 >> $c; fi
    local b=$(awk 'NF >= 3 && $2 + 0 >= bs {bs = $2 + 0; k = $1} END {print k}' $c)
    [ -n "$b" ] && { rm -rf $mp; mv $mp.$b $mp; }
    for k in g p m; do rm -rf $mp.$k; done
    echo "${b:-none} $(awk -v b=${b:-none} '$1 == b {print $2 + 0}' $c) ${cf:-0}"; }
  read -r wb ws wc <<< "$(unit $F $M nolocal)"; echo "whole $wb $ws confident $wc: $(tr '\n' ';' < $M.cand)" >> $M.choice
  echo "$F $M" > $M.units.new
  if [ "${OCT:-1}" = 1 ] && awk -v c=${wc:-0} -v t=${MINCONF:-2} 'BEGIN { exit !(c >= t) }' && awk -v s=${ws:-0} -v m=${wc:-0} -v f=$LM 'BEGIN { exit !(s < f * m) }'; then
    h=$(jq .h $F/fields.json); n2=$(( $(jq '.shape[0]' $F/fields.json) / 2 )); tot=0; : > $M.units.oct
    for k in 0 1 2 3 4 5 6 7; do oz=$(( (k >> 2 & 1) * n2 )) oy=$(( (k >> 1 & 1) * n2 )) ox=$(( (k & 1) * n2 )); fo=$F.o$k; rm -rf $fo; mkdir -p $fo
      for a in $F/*/; do ln -s $(readlink -f $a) $fo/$(basename $a); done
      jq --argjson z $oz --argjson y $oy --argjson x $ox --argjson n $n2 --argjson h $h '. + {origin: [.origin[0] + $z * $h, .origin[1] + $y * $h, .origin[2] + $x * $h],
          shape: [$n, $n, $n], zarr_offset: [$z, $y, $x], axis: (.axis[$z * $h : ($z + $n) * $h + 1] | map([.[0] - $y * $h, .[1] - $x * $h]))}' $F/fields.json > $fo/fields.json
      read -r ob os oc <<< "$(unit $fo $M.o$k)"; echo "o$k $ob $os confident $oc: $(tr '\n' ';' < $M.o$k.cand)" >> $M.choice
      [ "$ob" != none ] && echo "$fo $M.o$k" >> $M.units.oct; tot=$(awk -v t=$tot -v s=${os:-0} 'BEGIN {print t + s}')
    done
    om=$(awk -v t=$tot 'BEGIN {print t / 8}'); echo "octants mean $om vs whole ${ws:-0}" >> $M.choice
    awk -v a=$om -v b=${ws:-0} 'BEGIN { exit !(a > b) }' && mv $M.units.oct $M.units.new
  fi
  mv $M.units.new $M.units; touch $M.done; echo "$N: $(tail -n 1 $M.choice) units $(wc -l < $M.units) | $(head -1 $M.choice | cut -c1-60)";;
label)
  [ -e $M.done ] || { echo "$N: no fit"; exit 1; }
  SH=$(jq -c '.shape' $CT/$VOL/$LV/zarr.json | tr -d '[] '); DEV=(--gpu $G); [ "${CPU:-0}" = 1 ] && DEV=(--cpu --threads $TH)
  [ -s $M.units ] || echo "$F $M" > $M.units
  for arm in ${ARMS:-agree=0.3,0.15,0.3 corr=0.3,0.5,0.3 lo=0.15,0.15,0.3}; do nm=${arm%%=*}; gt=${arm#*=}
    [ -e $RD/stats/$N.$nm.txt ] && continue
    o=$RD/labels/$nm/$S; mkdir -p $o; : > $RD/stats/$N.$nm.tmp
    while read -r uf um; do   # (each unit writes its own 512^3 shards of the scan's label zarr)
      nice -n 5 env UFSM_FIT_THREADS=$TH $U fit $uf "${DEV[@]}" --model $um --winding $o --winding-mod14 3 --winding-step $STEP --winding-gate $gt \
        --winding-global $SH --phase-sign ${PSIGN:-1} ${CHECK:-} > $L.label-$nm.log 2>&1 || { echo "$N: label $nm failed: $(tail -1 $L.label-$nm.log)"; exit 1; }
      grep -E 'winding-gate|winding-check|^fit: dir' $L.label-$nm.log | sed "s#^#$(basename $um) #" >> $RD/stats/$N.$nm.tmp
    done < $M.units
    mv $RD/stats/$N.$nm.tmp $RD/stats/$N.$nm.txt
  done
  [ "${KEEPF:-1}" = 1 ] || rm -rf $F $F.o?
  echo "$N: $(for nm in $(ls $RD/stats | grep "^$N\." | sed "s/^$N\.//; s/\.txt$//"); do grep -h 'winding-gate' $RD/stats/$N.$nm.txt | sed -nE 's/.*\(([0-9.]+)%\).*/\1/p' | awk -v n=$nm '{s += $1; c++} END {printf "%s %.1f%% ", n, c ? s / c : 0}'; done)";;
*) echo "box.sh: stage pred | fit | label"; exit 2;;
esac

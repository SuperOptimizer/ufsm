#!/bin/bash
# gate.sh CKPT TAG (CAL, POD-S): COUNT's d1 gate (desk v2/count/dft/gate.sh + p139/eval.sh) with pod paths. Fields for 0139 + 1667 from CKPT,
# the pn2 whole-scroll ref (unroll.sh REFPH), wrap-check --gt-int (density / phase per GT turn by network spacing bin), fit --score-z
# per slab (ids, 0.5 / 4 mm pairs). Out: /root/cal/gate/<scroll>_<TAG>/{fields,pn2}. Summary: /root/cal/gate/<TAG>.txt
set -u; CK=$1; T=$2; C=/root/cal; U=${U:-/root/pod/ufsm/build/ufsm}; TH=${TH:-10}; O=$C/gate; mkdir -p $O
pred() {   # pred NAME CTKEY UM LEVEL BOX OCC
  local D=$O/$1_$T; mkdir -p $D; cp --update=none $C/ref/$1/axis.json $D/
  [ -e $D/fields/fields.json ] || { rm -rf $D/fields.tmp $D/fields; /usr/bin/time -v nice -n 5 $U predict $CK $C/ct $2 $D/fields.tmp --um $3 --level $4 --box $5 \
    --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 --grid-origin 0,0,0 --ema 1 --q 8 --threads $TH --gpu 0 --axis $D/axis.json --levels 1 \
    --fit-zarr $D/fields --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 --occupancy $6 > $D/pred.log 2>&1 && rm -rf $D/fields.tmp; }
  grep -E "Mvox|Elapsed|Maximum resident" $D/pred.log | tail -3
}
ev() {   # ev NAME CODES KEY
  local D=$O/$1_$T F=$O/$1_$T/fields P=$O/$1_$T/pn2; mkdir -p $P
  [ -e $P/ref/model.json ] || /usr/bin/time -v nice -n 5 $U count-field $F $P/cf.unused --phase-w 0.5 --wide 100,140 --wide-phase 15 --R 0 --pool 8 --phase-w 4 --iters 1200 \
    --model $P/ref --model-only 1 --model-stride 4 --threads $TH --mem-gb 16 > $P/cf.log 2>&1 || { echo "$1 cf failed"; return 1; }
  nice $U wrap-check $P/ref $F --codes $2 $3 --threads $TH --gt-int 1 --tsv $P/gti.tsv > $P/gti.log 2>&1
  local NZ=$(( $(jq -r ".shape[0]" $F/fields.json) * 2 )); local S=$(( NZ / 5 ))
  for i in 0 1 2 3 4; do nice $U fit $F --model $P/ref --gpu 0 --codes $2 $3 --score-z $((i * S)),$S --score-pairs 1 > $P/z$i.log 2>&1; done
}
summ() {   # summ NAME
  local P=$O/$1_$T/pn2
  echo "== $1 $T fit/GT $(awk -F'\t' '$1=="fit" {print $2, "off", $3}' $P/gti.tsv)  density/GT $(awk -F'\t' '$1=="density" {print $2, "off", $3}' $P/gti.tsv)"
  echo "   network spacing bins (density ratio/off, phase ratio/off):"; grep -A12 "network spacing" $P/gti.log | grep -E "^ +[0-9]|all" | sed 's/^/   /'
  for i in 0 1 2 3 4; do echo "   z$i $(grep -o "winding ids right [0-9.]*" $P/z$i.log) $(grep -o "0.5 mm [0-9.]*" $P/z$i.log | head -1) $(grep -o " 4 mm [0-9.]*" $P/z$i.log | head -1)"; done
}
pred p139 PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr 9.362 0 0,0,0,20544,6621,6621 3
pred p1667 PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr 2.399 2 0,0,0,9269,3136,3232 5
ev p139 $C/gt/PHerc0139-wind-20261016/codes 18.72
while [ ! -e $C/gt/PHerc1667-wind-20261016/codes.done ]; do sleep 60; done
ev p1667 $C/gt/PHerc1667-wind-20261016/codes 4.798
{ summ p139; summ p1667; } | tee $O/$T.txt

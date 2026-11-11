#!/usr/bin/env bash
# scripts/fibers/wbjoin_fib.sh FIBDIR OUT.tsv [SITES]: fibre votes in fit-join at the wbench Paris 4 slab sites, paired against the
# plain join. Per site (wbench.sh SLABS=1 protocol: fields FIELDS (default wind96), crops fA / fB / fC, production fits of A and B,
# interiors [32, 160) / [160, 288)), the two slab fits run once (kept: a rerun only joins); then fit-join per VARIANTS (name=flags;
# "base" = no fibres): fib<W>: --join-fib W, nearest pair ends (the votes then sample W at the fibre points next to the boundary,
# i.e. the overlap again); gap<W>: --join-fib-arc 1.2 --join-fib-gap 0.3 (both ends >= 0.3 mm = 31 native voxels from the boundary,
# beyond the slab overlap's edges: each slab's W in its own interior, carried along the fibre); *only: --chain-w 0, the fibres alone.
# Each joined set scored on fC against the Paris 4 codes (winding ids right, pair accuracy at 4 mm). Rows: site variant shift ids p4x.
set -u
FD=${1:?fibre dir}; OUT=${2:?out tsv}; SITES=${3:-4624 5392 5904 6672}; GPU=${GPU:-0}; FT=${FIELDS:-wind96}; VARIANTS=${VARIANTS:-base fib1 fib4 fibonly gap1 gap4 gaponly}
U=${U:-/vesuvius/ufsm/fibers/ufsm.fib3}   # (fit-join with --fibers)
UF=${UF:-/vesuvius/ufsm/wbench/bin/ufsm}  # (the slab fits and the joined scores: the pinned wbench fitter; this branch's fit stops with an
                                          #  illegal memory access after stage 4 on these 96 x 3328 x 3520 crops, 2026-11-11)
D=$(dirname "$(readlink -f "$0")"); THREADS=${THREADS:-8}
source $D/../wbench/site-desk.env
QR=$GT/paris4-q9p6/q-9p6.zarr; QK=9.6; L=$WB/gpu$GPU.lock; GF="--gpu $GPU"
g_ids() { grep -oE 'winding ids right [0-9.]+' $1 | tail -1 | awk '{print $4}'; }
g_pair() { grep -oE "pair winding right.*" $1 | grep -oE "$2 mm [0-9.na]+ cross [0-9.na]+" | awk -v c=$3 '{print (c ? $5 : $3)}'; }
[ -s $OUT ] || printf 'site\tvariant\tshift\tfib_votes\tfib_share\tids\tp4x\n' > $OUT
for s in $SITES; do
  S=site$s; w=$WB/runs/fibjoin/$S; mkdir -p $w
  f=$WB/fields/$FT/$S
  bash $D/../wbench/mkcrop.sh $f $w/fA 0 96 96 96 3328 3520 > /dev/null
  bash $D/../wbench/mkcrop.sh $f $w/fB 64 96 96 96 3328 3520 > /dev/null; bash $D/../wbench/mkcrop.sh $f $w/fC 0 96 96 160 3328 3520 > /dev/null
  for k in A B; do [ -e $w/fit$k.done ] && continue
    flock $L env UFSM_FIT_THREADS=$THREADS nice -n 5 $UF fit $w/f$k $GF --out $w/fit$k --codes $QR $QK --score-pairs 1 --dir 1 --uw-repair 8 > $w/fit$k.log 2>&1 && touch $w/fit$k.done; done
  until [ -s $FD/$S.fib ]; do sleep 60; done   # (the slab fits run while the fibres are traced)
  for v in $VARIANTS; do
    grep -qP "^$S\t$v\t" $OUT && continue
    G="--join-fib-arc 1.2 --join-fib-gap 0.3"
    case $v in base) J="";; fibonly) J="--chain-w 0 --fibers $FD/$S.fib --join-fib 1";; gaponly) J="--chain-w 0 --fibers $FD/$S.fib --join-fib 1 $G";;
      gap*) J="--fibers $FD/$S.fib --join-fib ${v#gap} $G";; *) J="--fibers $FD/$S.fib --join-fib ${v#fib}";; esac
    nice -n 5 $U fit-join $w/j_$v $w/fC --blend 16 $J $w/fitA 32 160 $w/fitB 160 288 > $w/join_$v.log 2>&1 || { echo "$S $v: join failed"; continue; }
    flock $L env UFSM_FIT_THREADS=$THREADS nice -n 5 $UF fit $w/fC --model $w/j_$v $GF --codes $QR $QK --score-pairs 1 > $w/joined_$v.log 2>&1
    sh=$(grep -E '^ +0\|1 ' $w/join_$v.log | tail -1 | awk '{print $4}'); fv=$(grep -oE '[0-9]+ votes' $w/join_$v.log | head -1 | awk '{print $1}')
    fs=$(grep -oE 'votes .residue <= 0.3 turn., mode [-+0-9]+ share [0-9.]+' $w/join_$v.log | awk '{print $NF}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $S $v "${sh:-na}" "${fv:-0}" "${fs:-na}" "$(g_ids $w/joined_$v.log)" "$(g_pair $w/joined_$v.log 4 1)" >> $OUT
    rm -rf $w/j_$v
  done
  echo "$S done"
done

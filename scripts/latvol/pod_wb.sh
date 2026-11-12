#!/bin/bash
# pod_wb.sh ARM [BOX...]: wbench rows (fit fields -> fit -> scores) of the checkpoint CK on the pod with the CT from the volcomp mirror
# (ARM ct) or from the latvol tree /workspace/latstore/ARM (ufsm latvol of the boxes' regions, the mirror's layout). One box per shared
# GPU unit (pod_shared.sh). Rows: /workspace/latstore/wb/res/u1-ARM.tsv, fields kept in wb/fields/u1-ARM/BOX.
ARM=${1:?arm}; shift
K=/workspace/ue; L=/workspace/latstore; H=$L/ufsm; CK=${CK:-/workspace/univI/u1b.ckpt}
BL=${*:-PH1 PH2 PH3 HOLD J1 J5 J9 J13 J19 J23 J27 J32 C9 C18 C30 C52}
mkdir -p $L/wb/bin; [ -e $L/wb/bin/ufsm ] || ln -s $(readlink -f $K/wb/bin/ufsm) $L/wb/bin/ufsm; [ -d $L/wb/ceil ] || cp -r $K/wb/ceil $L/wb/ceil
export SITE=podlv K WB=$L/wb GPU=0 SLABS=0 REP=${REP:-0} THREADS=${THREADS:-16} WZARGS= UP=$H/build/ufsm CACHE=$K/cache NOLOCK=1
[ "$ARM" != ct ] && export LVROOT=$L/$ARM
for b in $BL; do BOXES=$b bash $H/scripts/latvol/pod_shared.sh bash $H/scripts/wbench/wbench.sh $CK u1-$ARM 2>&1 | tail -2; done

#!/bin/bash
# pod_wb.sh ARM [BOX...]: wbench rows (fit fields -> fit -> scores) of the checkpoint CK (tag TAG) on the pod with the CT from the volcomp
# mirror (ARM ct) or from the tree $LVBASE/ARM (ufsm latvol of the boxes' regions, the mirror's layout: latent cells or re-encoded
# volcomp). One box per shared GPU unit (pod_shared.sh). Rows: /workspace/latstore/wb/res/TAG-ARM.tsv. Latent-grid checkpoints (u1,
# wz896): LATGRID=1 (cubic windows, wbench's WZARGS= ).
ARM=${1:?arm}; shift
K=/workspace/ue; L=/workspace/latstore; H=$L/ufsm; CK=${CK:-/workspace/univI/u1b.ckpt}; TAG=${TAG:-u1}
BL=${*:-PH1 PH2 PH3 HOLD J1 J5 J9 J13 J19 J23 J27 J32 C9 C18 C30 C52}
mkdir -p $L/wb/bin; [ -e $L/wb/bin/ufsm ] || ln -s $(readlink -f $K/wb/bin/ufsm) $L/wb/bin/ufsm; [ -d $L/wb/ceil ] || cp -r $K/wb/ceil $L/wb/ceil
export SITE=podlv K WB=$L/wb GPU=0 SLABS=0 REP=${REP:-0} THREADS=${THREADS:-16} UP=$H/build/ufsm CACHE=$K/cache NOLOCK=1 SHN=latstore-$TAG
[ "${LATGRID:-1}" = 1 ] && export WZARGS=
[ "$ARM" != ct ] && export LVROOT=${LVBASE:-$L}/$ARM
for b in $BL; do BOXES=$b bash $H/scripts/latvol/pod_shared.sh bash $H/scripts/wbench/wbench.sh $CK $TAG-$ARM 2>&1 | tail -2; done
# the fields' checksums per box (an arm whose fields equal another arm's saw the same input: void), then the fields go
for b in $BL; do [ -d $L/wb/fields/$TAG-$ARM/$b ] && echo "$TAG $ARM $b $(cd $L/wb/fields/$TAG-$ARM/$b && find . -type f ! -name '*.json' | sort | xargs cat | md5sum | cut -c1-16)"; done >> $L/wb/fields.md5
[ "${KEEPF:-0}" = 1 ] || rm -rf $L/wb/fields/$TAG-$ARM

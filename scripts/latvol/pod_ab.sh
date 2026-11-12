#!/bin/bash
# pod_ab.sh: the latent-store accuracy A/B on the pod: latvol trees of the wbench boxes (C 29 / 64, steps 0.25 / 0.354 / 0.5) and the
# wbench rows of CK from each against the volcomp CT. Logs under /workspace/latstore/ab/.
L=/workspace/latstore; H=$L/ufsm; U=$H/build/ufsm; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; mkdir -p $L/ab
BX="PH1 PH2 PH3 HOLD J1 J5 J9 J13 J19 J23 J27 J32 C9 C18 C30 C52"
P4=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr; P7=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr
bash $H/scripts/latvol/wb_boxes.sh paris4 $BX > $L/ab/p4.boxes; bash $H/scripts/latvol/wb_boxes.sh 1667 $BX > $L/ab/p7.boxes
enc() {   # NAME C STEP
  for kv in "$P4 p4" "$P7 p7"; do set -- $1 $2 $3 $kv
    [ -e $L/$1/$4/2/.done ] && continue; mkdir -p $L/$1/$(dirname $4)
    bash $H/scripts/latvol/pod_shared.sh $U latvol $L/klt1.lcb $M $4 $L/$1/$4 --levels 2 --gpu 0 --threads 32 --cache $L/cache --c $2 --step $3 --boxes $L/ab/$5.boxes > $L/ab/enc_$1_$5.log 2>&1 && touch $L/$1/$4/2/.done
    tr '\r' '\n' < $L/ab/enc_$1_$5.log | grep -E "sample boxes|bytes per voxel; " | cut -c1-420
  done
}
for arm in ${ARMS:-ct lv29 lv64 lv29s35 lv29s50}; do
  case $arm in lv29) enc lv29 29 0.25;; lv64) enc lv64 64 0.25;; lv29s35) enc lv29s35 29 0.354;; lv29s50) enc lv29s50 29 0.5;; esac
  bash $H/scripts/latvol/pod_wb.sh $arm $BX > $L/ab/wb_$arm.log 2>&1
  echo "arm $arm done $(date +%T)"
done

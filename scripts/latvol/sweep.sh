#!/bin/bash
# sweep.sh ARM...: the compression sweep's inference part on the pod (DESIGN 2026-11-12). Per ARM: the wbench boxes' CT re-encoded into
# /workspace/latstore/sw/ARM, then the wbench rows of the three winding checkpoints from it (in parallel, shared GPU units).
#   orig        the upstream uncompressed CT (public bucket, zarr v2) as lossless volcomp: the top anchor and every other arm's source
#   vqQ         volcomp at quality Q                      (vq2: the mirror's setting for the 2.4 um scans' level 2; vq8: 0139's level 0)
#   lvC_S       latent cells, C coefficients, step S x sd (lv29_0.25: u1's own quantiser)
# Rates: sw/rates.tsv (arm scan bytes voxels). Rows: wb/res/{u1,wz896,wind96}-ARM.tsv
L=/workspace/latstore; H=$L/ufsm; U=$H/build/ufsm; SW=$L/sw; mkdir -p $SW/log
BK=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
BX=${BX:-PH1 PH2 PH3 HOLD J1 J9 J19 J27 C9 C18 C30 C52 139-5 139-2}
SCANS="PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr:paris4:2 PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr:1667:2 PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr:PHerc0139:0"
enc() {   # ARM
  local a=$1 s key sc lv args src
  for s in $SCANS; do IFS=: read -r key sc lv <<< "$s"
    [ -e $SW/$a/$key/.done ] && continue
    bash $H/scripts/latvol/wb_boxes.sh $sc $BX > $SW/log/$sc.boxes; rm -rf $SW/$a/$key; mkdir -p $SW/$a/$(dirname $key)
    case $a in
      orig) src="$BK $key"; args="--zarr2 1 --group-from $M --volcomp 0";;
      vq*)  src="$SW/orig $key"; args="--volcomp ${a#vq}";;
      lv*)  c=${a#lv}; src="$SW/orig $key"; args="--c ${c%_*} --step ${c#*_} --gpu -1";;
    esac
    UFSM_LATVOL_SAMPLE_BOXES=1 nice $U latvol $L/klt1.lcb $src $SW/$a/$key --levels $lv --threads ${ETH:-24} --boxes $SW/log/$sc.boxes $args > $SW/log/enc_${a}_$sc.log 2>&1 || { echo "encode $a $sc failed"; return 1; }
    grep -a '^rate:' $SW/log/enc_${a}_$sc.log | awk -v a=$a -v s=$sc '{print a "\t" s "\t" $4 "\t" $7}' >> $SW/rates.tsv; touch $SW/$a/$key/.done
  done
}
for a in "$@"; do
  enc $a || continue
  [ "${ENCONLY:-0}" = 1 ] && continue
  ( CK=${CKU1:-$L/u1-8000.ckpt} TAG=u1 LATGRID=1 LVBASE=$SW bash $H/scripts/latvol/pod_wb.sh $a $BX > $SW/log/wb_u1_$a.log 2>&1 ) &
  ( CK=/workspace/ue/ckpt/lat-wz896p.ckpt TAG=wz896 LATGRID=1 LVBASE=$SW bash $H/scripts/latvol/pod_wb.sh $a $BX > $SW/log/wb_wz896_$a.log 2>&1 ) &
  ( CK=/workspace/ue/ckpt/wind96p.ckpt TAG=wind96 LATGRID=0 LVBASE=$SW bash $H/scripts/latvol/pod_wb.sh $a $BX > $SW/log/wb_wind96_$a.log 2>&1 ) &
  wait; echo "arm $a done $(date +%T)"
done

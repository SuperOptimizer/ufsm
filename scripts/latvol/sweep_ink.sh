#!/bin/bash
# sweep_ink.sh ARM...: the compression sweep's ink score on the pod: the ink3d-ext teacher on a subset of the held-out crops with the
# level-0 CT of the crop boxes re-encoded per arm (orig / vqQ / lvC_S as sweep.sh, trees under /workspace/latstore/ski/ARM), scored by
# scripts/univ/eval.sh TASKS=ink (2-D ink-score + voxelwise ink-vscore). One crop = one shared GPU unit. Results:
# /workspace/latstore/evi/ARM/ink3dext-ARM/ink/{scores,vscores}.tsv; rates ski/rates.tsv.
L=/workspace/latstore; H=$L/ufsm; U=$H/build/ufsm; SW=$L/ski; mkdir -p $SW/log
BK=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
CR=${CR:-w07_1 w05_1 PHerc0139_w044_1 PHerc1667_w029_0}; P4=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
A=/workspace/ue/wb/inkeval/crops; B=/workspace/ue/inkgt/evcrops
# the crops' boxes per CT key (level-0 voxels, a 128 margin)
rm -f $SW/log/*.boxes; for c in $CR; do
  l=$(grep -hP "^$c\t" $A/crops.tsv $B/crops.tsv | head -1); box=$(echo "$l" | cut -f5); key=$(echo "$l" | cut -f7); key=${key:-$P4}
  echo $box | awk -F, '{print $1 - 128, $2 - 128, $3 - 128, $4 + 256, $5 + 256, $6 + 256}' >> $SW/log/$(echo $key | tr / _).boxes; echo $key; done | sort -u > $SW/log/keys
enc() {
  local a=$1 key args src
  while read -r key; do [ -e $SW/$a/$key/.done ] && continue; bf=$SW/log/$(echo $key | tr / _).boxes; rm -rf $SW/$a/$key; mkdir -p $SW/$a/$(dirname $key)
    case $a in
      orig) src="$BK $key"; args="--zarr2 1 --group-from $M --volcomp 0";;
      vq*)  src="$SW/orig $key"; args="--volcomp ${a#vq}";;
      lv*)  c=${a#lv}; src="$SW/orig $key"; args="--c ${c%_*} --step ${c#*_} --gpu -1";;
    esac
    UFSM_LATVOL_SAMPLE_BOXES=1 nice $U latvol $L/klt1.lcb $src $SW/$a/$key --levels 0 --threads ${ETH:-24} --boxes $bf $args > $SW/log/enc_${a}_$(basename $key).log 2>&1 || { echo "encode $a $key failed"; return 1; }
    grep -a '^rate:' $SW/log/enc_${a}_$(basename $key).log | awk -v a=$a -v s=$(basename $key) '{print a "\t" s "\t" $4 "\t" $7}' >> $SW/rates.tsv; touch $SW/$a/$key/.done
  done < $SW/log/keys
}
for a in "$@"; do
  enc $a || continue; [ "${ENCONLY:-0}" = 1 ] && continue
  WB=$L/iwb/$a; E=$L/evi/$a; mkdir -p $WB/res $WB/inkeval/crops $WB/evc $E
  for c in $CR; do
    : > $WB/inkeval/crops/crops.tsv; : > $WB/evc/crops.tsv
    if grep -qP "^$c\t" $A/crops.tsv; then grep -P "^$c\t" $A/crops.tsv > $WB/inkeval/crops/crops.tsv; ln -sfn $A/$c $WB/inkeval/crops/$c
    else grep -P "^$c\t" $B/crops.tsv > $WB/evc/crops.tsv; ln -sfn $B/$c $WB/evc/$c; fi
    SHN=latstore-ink bash $H/scripts/latvol/pod_shared.sh env SITE=pod K=/workspace/ue WB=$WB E=$E CACHE=$L/cache/ink-$a GPU=0 U=$U CTROOT=$SW/$a TASKS=ink INKP=ink3dext \
      INKW=/workspace/ue/ckpt/ckpt_78k_fullsup.pth INKC=$WB/evc INKGT=/workspace/ue/inkgt/evgt bash $H/scripts/univ/eval.sh /workspace/ue/ckpt/ink3dext.ckpt ink3dext-$a >> $SW/log/ink_$a.log 2>&1
  done
  echo "arm $a done $(date +%T): $(cut -f1,8,9 $E/ink3dext-$a/ink/vscores.tsv 2>/dev/null | tr '\n\t' '  ')"; rm -rf $E/ink3dext-$a/ink/vol_* $E/ink3dext-$a/ink/surf
done

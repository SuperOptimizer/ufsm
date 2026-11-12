#!/bin/bash
# scripts/latvol/rate20.sh encode | score CKPT TAG: the latent store's standard input setting at ~20x vs 8-bit (0.050 +- 0.005 B/vox; user
# decision 2026-11-12), on the desk. sweep.sh's method (DESIGN "How far the CT can be compressed"): the 14 wbench boxes' regions of the
# upstream uncompressed CT (public bucket, zarr v2) kept as lossless volcomp (arm orig, the anchor), every candidate encoded from it with
# ufsm latvol (latent cells lvC_S: C coefficients of klt1, step S x sd), then wbench rows (REP=0, as the sweep) per arm, paired per box
# against orig. Candidates (ARMS): lv29_S (S chosen for the rate), lv20_1.0, lv13_1.0 (sweep A's arm, the anchor of the rerun).
#   encode   CPU only (--gpu -1), resumable per arm and scan; rates -> $SW/rates.tsv (arm scroll bytes voxels)
#   score    one GPU window: rows TAG-ARM for orig + each arm (GPUS, two queues), then the paired table vs TAG-orig
set -u
MODE=${1:?encode|score}; H=${H:-$(cd "$(dirname "$0")/../.." && pwd)}; U=${ENC:-/home/forrest/ufsm-lvm/build/ufsm}; BASIS=/vesuvius/ufsm/v2/lat/codec/klt1.lcb
SW=${SW:-/mnt/scratch/latvol/rate20}; mkdir -p $SW/log; WB=/vesuvius/ufsm/wbench
BK=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
BX=${BX:-PH1 PH2 PH3 HOLD J1 J9 J19 J27 C9 C18 C30 C52 139-5 139-2}
SCANS="PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr:paris4:2 PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr:1667:2 PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr:PHerc0139:0"
ARMS=${ARMS:-lv29_1.5 lv20_1.25 lv13_1.0}
MC=""; [ -x ~/bin/memcap ] && MC="$HOME/bin/memcap -m ${ENCMEM:-24G}"
enc() {   # ARM
  local a=$1 s key sc lv args src c
  for s in $SCANS; do IFS=: read -r key sc lv <<< "$s"
    [ -e $SW/$a/$key/.done ] && continue
    bash $H/scripts/latvol/wb_boxes.sh $sc $BX > $SW/log/$sc.boxes; [ -s $SW/log/$sc.boxes ] || continue
    rm -rf $SW/$a/$key; mkdir -p $SW/$a/$(dirname $key)
    case $a in
      orig) src="$BK $key"; args="--zarr2 1 --group-from $M --volcomp 0";;
      lv*)  c=${a#lv}; src="$SW/orig $key"; args="--c ${c%_*} --step ${c#*_} --gpu -1";;
    esac
    UFSM_LATVOL_SAMPLE_BOXES=1 nice -n 15 $MC $U latvol $BASIS $src $SW/$a/$key --levels $lv --threads ${ETH:-12} --boxes $SW/log/$sc.boxes $args > $SW/log/enc_${a}_$sc.log 2>&1 || { echo "encode $a $sc failed (see $SW/log/enc_${a}_$sc.log)"; return 1; }
    grep -a '^rate:' $SW/log/enc_${a}_$sc.log | awk -v a=$a -v s=$sc '{print a "\t" s "\t" $4 "\t" $7}' >> $SW/rates.tsv; touch $SW/$a/$key/.done
  done
}
rate() { awk -F'\t' -v a=$1 '$1 == a {b += $3; v += $4} END {if (v) printf "%.4f", b / v}' $SW/rates.tsv; }
case $MODE in
  encode)
    enc orig || exit 1; echo "orig: $(rate orig) B/vox"
    for a in $ARMS; do enc $a && echo "$a: $(rate $a) B/vox"; done;;
  scorearm|table|score)
    CK=$(readlink -f ${2:?ckpt}); T=${3:?tag}; UP=${UP:-/home/forrest/ufsm-eval6b7/build/ufsm}; GPUS=${GPUS:-0 1}
    # wbench on the arm's tree: LVROOT replaces the mirror and the Paris 4 root (keys with volumes/, the mirror's layout)
    run() { local a=$1 g=$2; env SITE=desk WB=$WB GPU=$g SLABS=0 REP=0 THREADS=12 PRIO=eval UP=$UP WZARGS= WIN=640 HALO=64 LVROOT=$SW/$a \
        P4V=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr BOXES="$BX" $MC bash $H/scripts/wbench/wbench.sh $CK $T-$a > $SW/log/wb_${T}_$a.log 2>&1; }
    tab() { { echo "== $T: the ~20x input setting (0.050 +- 0.005 B/vox), paired vs $T-orig (the uncompressed anchor, $(rate orig) B/vox), REP=0, 14 boxes"
      for a in $ARMS; do echo "-- $a  $(rate $a) B/vox"; $U wbench-stats $WB/res/$T-$a.tsv $WB/res/$T-orig.tsv --metric p05x,ids 2>&1 | grep -E '^(p05x|ids) '; done; } | tee $SW/$T.table.txt; }
    # scorearm CK TAG ARM (GPU): one arm's rows (a GPU-window job of deskgate.sh); table CK TAG: the paired table from the rows
    if [ $MODE = scorearm ]; then a=${4:?arm}; [ -e $SW/$a/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/.done ] || { echo "rate20: $a not encoded"; exit 1; }
      run $a ${GPU:-0}; exit; fi
    [ $MODE = table ] && { tab; exit; }
    set -- $GPUS; i=0; for a in orig $ARMS; do g=${@:$(( i % $# + 1 )):1}; i=$((i + 1)); echo "$a $g"; done > $SW/log/plan_$T
    for g in $GPUS; do ( awk -v g=$g '$2 == g {print $1}' $SW/log/plan_$T | while read -r a; do run $a $g; done ) & done; wait
    tab;;
esac

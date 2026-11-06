#!/bin/bash
# (copy of the desk kit /mnt/scratch/v2/trace/ev.sh; needs the GT boxes there: <BOX>/{prob,gt}.raw)
# ev.sh BOX "TRACE ARGS" "SOLVE ARGS" [TAG]: sheet-trace + patch-score + patch-solve on a GT box; one summary line
# boxes: 1667 C*, 0139 139-*, P4 HOLD
T=/mnt/scratch/v2/trace; U=${U:-/vesuvius/ufsm/v3/ps-src/build/ufsm}; B=$1; TA=$2; SA=$3; TAG=${4:-x}
case $B in
  C2) O=3328,1152,512;; C3) O=5120,1024,768;; C7) O=3840,2304,1152;; C1) O=7168,1024,896;; C4) O=1024,1280,2304;; C9) O=1280,1152,896;;
  C11) O=4864,768,1408;; C19) O=384,1280,2432;; C21) O=7808,2688,1280;; C24) O=3200,1792,896;;
  HOLD) O=12288,4096,3584;; 139-1) O=3584,3584,1536;; 139-2) O=5376,4096,2560;; 139-4) O=5120,2304,2560;; 139-5) O=5376,3072,3840;;
esac
case $B in
  C*) AX=/mnt/scratch/pods/PHerc1667/unroll/axis.json; AXS=0.25; SGN=-1; VS=9.596; N=512,512,512;;
  HOLD) AX=/vesuvius/ufsm/paris4/ckpt/axis-0.json; AXS=0.25; SGN=${SGNP4:-1}; VS=9.6; N=512,576,768;;
  139-*) AX=/vesuvius/ufsm/gt/axis/PHerc0139-umb-scan.json; AXS=1; SGN=${SGN139:-1}; VS=9.362; N=512,512,512;;
esac
BX=$O,$N; W=$T/ev/$B.$TAG; rm -rf $W; mkdir -p $W
$U sheet-trace --prob $T/$B/prob.raw --box $BX --axis $AX --axis-scale $AXS --out $W --vs $VS --threads 8 $TA > $W/trace.txt
sc=$($U patch-score --paths $W/paths --gt $T/$B/gt.raw --box $BX --axis $AX --axis-scale $AXS --sgn $SGN | sed -E 's/.*recall ([0-9.]+) \| precision ([0-9.]+).*purity \(matched-weighted\) ([0-9.]+).*/\1 \2 \3/')
so=$($U patch-solve --paths $W/paths --prob $T/$B/prob.raw --gt $T/$B/gt.raw --box $BX --axis $AX --axis-scale $AXS --sgn $SGN --threads 8 --cut 0 --minv 32 --links $W/links.tsv --frame $W/frame.txt --tsv $W/nodes.tsv $SA > $W/solve.txt; grep "^sign +1" $W/solve.txt | sed -E 's/.*turn ([0-9.]+).*/\1/'; grep "purity ceiling" $W/solve.txt | sed -E 's/.*share\) ([0-9.]+)/\1/')
set -- $sc $so
printf "%-6s %-4s rec %s prec %s pur %s | solve %s ceil %s | eff %.3f | %s | %s\n" $B $TAG $1 $2 $3 $4 $5 $(echo "$1 * $4" | bc -l) "$TA" "$SA"

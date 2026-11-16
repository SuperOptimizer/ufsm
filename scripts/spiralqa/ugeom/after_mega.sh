#!/usr/bin/env bash
# after_mega.sh STEP: the audit ugeom measurements that need a GPU, queued for after the mega test pass (run in a GPU window).
#   filled  - 1667 unrolled from pub1667's filled store (the 2023 7.91 um scan outside a >= 1 mm dilated mask; CTROOT / CTKEY env) up to
#             snap, then pubcmp.sh: expect "B1 on zero (masked) CT" ~0 and cov50 78.5 -> ~90 %; snap.log's masked share < 0.02
#   wide    - count-field ref variants on 1667 (REFX), each scored on the 70 held-out C boxes (gtboxes.sh, CPU): the ref makes every
#             integer error (ref-cf ids 0.840 clean); does a confidence weighting of wide-spacing / weak-phase cells lift it?
#   packed  - Paris 4's packed boxes (ref ids 0.12-0.21, |err| median ~0.24 = random; 75 um pitch = ~4 field cells of 19.2 um per sheet):
#             count-field ref variants that re-weight the density where the network spacing is small (--rho-pow A,S0, --rho-scale),
#             scored on PH1-3 / HOLD (q-9p6) and the 34 July J boxes (codesB48); plus the drift (cf-demod) of each
#   demod   - unroll.sh DEMOD=1 on 1667 and P4 (no slabs: the ref + cf-demod) vs the slab run: GT boxes + pubcmp + strip length / time
set -euo pipefail
STEP=${1:?filled|wide|packed|demod}; H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$H/../../.." && pwd)
U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}; R1667=/vesuvius/ufsm/unroll/PHerc1667-u1; OUT=${OUT:-/vesuvius/ufsm/audit-20261114/ugeom/after}
BX=$R1667/score/boxes.tsv; GT="/vesuvius/ufsm/gt/PHerc1667-wind-20261016/codes 4.798"
mkdir -p "$OUT"
# (the spiral hand from spiral_dir.tsv: count-field refuses an evidence-free direction since 11-15, spiral-hand fix)
sdir() { awk -F'\t' -v s="$1" '$1 == s && $2 ~ /^[+-]?1$/ { print $2 + 0; exit }' "$REPO/scripts/unroll/spiral_dir.tsv"; }
case $STEP in
filled)
  : "${CTROOT:?the filled store root}" "${CTKEY:?the filled store key}"
  TAG=-filled UPTO=snap CK=${CK:-/vesuvius/ufsm/v2/lat/runs/u1/snap/13000.ckpt} CTROOT=$CTROOT CTKEY=$CTKEY bash "$REPO/scripts/unroll/unroll.sh" PHerc1667
  bash "$H/pubcmp.sh" /vesuvius/ufsm/unroll/PHerc1667-filled "$OUT/pub-filled" ;;
wide)
  for v in "base|" "wd5|--wide-dens 0.5" "wd9|--wide-dens 0.9" "pw2|--phase-w 2"; do
    n=${v%%|*} x=${v#*|}; d=$OUT/ref-$n
    [ -e $d/model.json ] || CUDA_VISIBLE_DEVICES=${GPU:-0} nice -n 10 $U count-field $R1667/fields/lat $OUT/cf.unused --dir $(sdir PHerc1667) --R 0 --pool 8 --phase-w 0.5 --wide 100,140 --wide-phase 15 \
      --model-only 1 --model-stride 4 $x --iters 1200 --model $d --threads 20 --mem-gb 24 > $OUT/ref-$n.log 2>&1
    bash "$H/gtboxes.sh" $R1667/fields/lat $d $BX $GT $OUT/gt-ref-$n $U | tee -a $OUT/wide.txt
  done ;;
packed)
  P4=/vesuvius/ufsm/unroll/PHercParis4-u1; PF=$P4/fields/lat; [ -e $P4/fields/rsp/fields.json ] && PF=$P4/fields/rsp
  awk -F'\t' '$2 == "paris4"' /vesuvius/ufsm/wbench/boxes.tsv > $OUT/p4boxes.tsv; grep -v '^#' /vesuvius/ufsm/wbench/cand-p4j.tsv > $OUT/p4j.tsv
  for v in "base|" "rp2|--rho-pow 0.2,150" "rp4|--rho-pow 0.4,150" "rs12|--rho-scale 1.2" "pw4|--phase-w 4"; do
    n=${v%%|*} x=${v#*|}; d=$OUT/p4ref-$n
    [ -e $d/model.json ] || CUDA_VISIBLE_DEVICES=${GPU:-0} nice -n 10 $U count-field $PF $OUT/cf.unused --dir $(sdir PHercParis4) --R 0 --pool 8 --phase-w 0.5 --wide 100,140 --wide-phase 15 \
      --model-only 1 --model-stride 4 $x --iters 1200 --model $d --threads 20 --mem-gb 24 > $OUT/p4ref-$n.log 2>&1
    bash "$H/gtboxes.sh" $PF $d $OUT/p4boxes.tsv /vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr 9.6 $OUT/gt-p4-$n $U | sed "s/^/$n PH: /" | tee -a $OUT/packed.txt
    bash "$H/gtboxes.sh" $PF $d $OUT/p4j.tsv /vesuvius/ufsm/datafix-20261114/raster/p4-codesB48 4.8 $OUT/gtj-p4-$n $U | sed "s/^/$n J: /" | tee -a $OUT/packed.txt
    nice -n 10 $U cf-demod $d $PF --step 8 --band 128 --cz 16 --threads 20 | grep 'z drift' | sed "s/^/$n /" | tee -a $OUT/packed.txt
  done ;;
demod)
  for S in PHerc1667 PHercParis4; do TAG=-demod DEMOD=1 UPTO=snap bash "$REPO/scripts/unroll/unroll.sh" $S; done
  bash "$H/gtboxes.sh" $R1667/fields/lat /vesuvius/ufsm/unroll/PHerc1667-demod/joined $BX $GT $OUT/gt-demod-1667 $U
  bash "$H/pubcmp.sh" /vesuvius/ufsm/unroll/PHerc1667-demod "$OUT/pub-demod" ;;
esac

#!/usr/bin/env bash
# after_mega.sh STEP: the audit ugeom measurements that need a GPU, queued for after the mega test pass (run in a GPU window).
#   filled  - 1667 unrolled from pub1667's filled store (the 2023 7.91 um scan outside a >= 1 mm dilated mask; CTROOT / CTKEY env) up to
#             snap, then pubcmp.sh: expect "B1 on zero (masked) CT" ~0 and cov50 78.5 -> ~90 %; snap.log's masked share < 0.02
#   wide    - count-field ref variants on 1667 (REFX), each scored on the 70 held-out C boxes (gtboxes.sh, CPU): the ref makes every
#             integer error (ref-cf ids 0.840 clean); does a confidence weighting of wide-spacing / weak-phase cells lift it?
#   demod   - unroll.sh DEMOD=1 on 1667 and P4 (no slabs: the ref + cf-demod) vs the slab run: GT boxes + pubcmp + strip length / time
set -euo pipefail
STEP=${1:?filled|wide|demod}; H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$H/../../.." && pwd)
U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}; R1667=/vesuvius/ufsm/unroll/PHerc1667-u1; OUT=${OUT:-/vesuvius/ufsm/audit-20261114/ugeom/after}
BX=$R1667/score/boxes.tsv; GT="/vesuvius/ufsm/gt/PHerc1667-wind-20261016/codes 4.798"
mkdir -p "$OUT"
case $STEP in
filled)
  : "${CTROOT:?the filled store root}" "${CTKEY:?the filled store key}"
  TAG=-filled UPTO=snap CK=${CK:-/vesuvius/ufsm/v2/lat/runs/u1/snap/13000.ckpt} CTROOT=$CTROOT CTKEY=$CTKEY bash "$REPO/scripts/unroll/unroll.sh" PHerc1667
  bash "$H/pubcmp.sh" /vesuvius/ufsm/unroll/PHerc1667-filled "$OUT/pub-filled" ;;
wide)
  for v in "base|" "wd5|--wide-dens 0.5" "wd9|--wide-dens 0.9" "pw2|--phase-w 2"; do
    n=${v%%|*} x=${v#*|}; d=$OUT/ref-$n
    [ -e $d/model.json ] || nice -n 10 $U count-field $R1667/fields/lat $OUT/cf.unused --R 0 --pool 8 --phase-w 0.5 --wide 100,140 --wide-phase 15 \
      --model-only 1 --model-stride 4 $x --iters 1200 --model $d --threads 20 --mem-gb 24 > $OUT/ref-$n.log 2>&1
    bash "$H/gtboxes.sh" $R1667/fields/lat $d $BX $GT $OUT/gt-ref-$n $U | tee -a $OUT/wide.txt
  done ;;
demod)
  for S in PHerc1667 PHercParis4; do TAG=-demod DEMOD=1 UPTO=snap bash "$REPO/scripts/unroll/unroll.sh" $S; done
  bash "$H/gtboxes.sh" $R1667/fields/lat /vesuvius/ufsm/unroll/PHerc1667-demod/joined $BX $GT $OUT/gt-demod-1667 $U
  bash "$H/pubcmp.sh" /vesuvius/ufsm/unroll/PHerc1667-demod "$OUT/pub-demod" ;;
esac

#!/usr/bin/env bash
# v2 universal model evaluation (plan bright-forging-seahorse, Phase 2). Usage: eval.sh MODE CKPT TAG
#   MODE screen (laptop GPU, under gpuq / gpu.lock as the called scripts take it):
#     1. winding: scripts/v2/wfields.sh CKPT v2-TAG (laptop predicts, desk fits: no ufsm fit on the laptop) -> wgate (R1 p05x CI > 0 and ids >= 0, R2 slab sites, R3 1667 guard vs wind96; the slab
#        sites run on forlindesk as wgate does from the laptop). The candidate's fields come from its unwrap2 channels 0-10.
#     2. ink: scripts/v2/inkev.sh with PX="--channel ink" (holdout F1 0139 w039 9.36 / 2.4, 1667 w028, Paris 4 segment box)
#        -> ~/ufsm-data/v2eval/ink/eval.tsv row v2-TAG; compare with the ift3-24000 row.
#   MODE milestone (desk, GPU 0 unless GPUS; only once an arm passes the screens; writes /vesuvius/ufsm/v2/unroll only):
#     1. whole Paris 4 through the c2f pipeline (scripts/unroll/unroll.sh) with the candidate as fields AND ink checkpoint
#        (INKPX="--channel ink"), TAG suffix -v2TAG; GT score fit --model joined --codes q-9p6 9.6 --score-pairs 1 vs .1626 / .681 / .303;
#        ray-check (human relative windings) on the candidate's and wind96's whole fields, paired.
#     2. held-out whole scrolls (HOLD, default "PHerc1667 PHerc0139 PHerc1299 PHerc0846A"): the same unroll, then join / wrap-check /
#        on-papyrus (surf-stats --ct-only) / ink-cover vs the current unroll of that scroll (CUR_<scroll> or $CURROOT/<scroll>), wraps
#        matched by wrap-match. Renders under <run>/strip/img_fill: LOOK at them before reporting.
#   MODE p4 / hold: only that half of milestone.
# Env: U (desk binary, default /vesuvius/ufsm/v2/bin/ufsm), UR (/vesuvius/ufsm/v2/unroll), REFUR (/vesuvius/ufsm/unroll: wind96 runs),
#   GT (q-9p6), RCW (relative_windings.json), FIELDGPUS / FITGPUS / INKGPU (unroll.sh; default all on GPU 0 when the other trains).
set -u
MODE=${1:?usage: eval.sh screen|milestone|p4|hold CKPT TAG}; CK=$(readlink -f "${2:?CKPT}"); TAG=${3:?TAG}
H=$(cd "$(dirname "$0")" && pwd); R=$(cd $H/../.. && pwd)
log() { echo "$(date '+%F %T') eval $TAG: $*"; }
screen() {
  log "winding gate vs wind96 (wfields.sh: laptop predicts the box / site fields, the desk fits them: wgate FIXED=1)"
  bash $H/wfields.sh $CK v2-$TAG || log "wfields failed"
  log "ink holdout F1 (inkev.sh, --channel ink)"
  PX="--channel ink" bash $H/inkev.sh $CK v2-$TAG || log "ink eval failed"
  log "done: wgate table desk /vesuvius/ufsm/wbench/gate/v2-$TAG/table.txt, ink row v2-$TAG in ~/ufsm-data/v2eval/ink/eval.tsv (baseline row ift3-24000)"
}
V=/vesuvius/ufsm/v2; U=${U:-$V/bin/ufsm}; UR=${UR:-$V/unroll}; REFUR=${REFUR:-/vesuvius/ufsm/unroll}
GT=${GT:-/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr}; RCW=${RCW:-$V/eval/relative_windings.json}
unroll() {   # unroll SCROLL: the candidate as fields + ink net (one model), -> $UR/SCROLL-v2TAG
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); [ "$avail" -ge 40 ] || { log "MemAvailable $avail GB < 40: wait"; return 1; }
  PERWRAP=${PERWRAP:-1} CK=$CK INKCK=$CK INKPX="--channel ink" TAG=-v2$TAG UR=$UR U=$U FIELDGPUS="${FIELDGPUS:-0}" FITGPUS="${FITGPUS:-0}" INKGPU=${INKGPU:-0} \
    bash $R/scripts/unroll/unroll.sh $1
}
p4() {
  local O=$UR/PHercParis4-v2$TAG F
  unroll PHercParis4 || { log "Paris 4 unroll failed ($O/progress.log)"; return 1; }
  F=$(ls -d $O/fields/*/ 2>/dev/null | head -1)
  $U fit ${F%/} --model $O/joined --codes $GT 9.6 --score-pairs 1 > $O/gt.log 2>&1
  log "Paris 4 GT (baseline .1626 ids / .681 / .303 pairs): $(grep -iE 'ids|pairs' $O/gt.log | tail -3 | tr '\n' ' ')"
  if [ -s $RCW ]; then
    $U ray-check $RCW ${F%/} --tsv $O/rc.tsv > $O/rc.log 2>&1
    RF=$(ls -d $REFUR/PHercParis4/fields/*/ 2>/dev/null | head -1)
    [ -s $V/eval/rc-w96.tsv ] || { [ -n "$RF" ] && $U ray-check $RCW ${RF%/} --tsv $V/eval/rc-w96.tsv > $V/eval/rc-w96.log 2>&1; }
    log "ray-check: $(grep -iE 'off|>= *0.5' $O/rc.log | head -3 | tr '\n' ' ') (wind96: $(grep -iE 'off|>= *0.5' $V/eval/rc-w96.log 2>/dev/null | head -3 | tr '\n' ' '))"
  else log "ray-check skipped: no $RCW (copy ~/ufsm-data/villawind/spiral/relative_windings.json there)"; fi
}
hold() {
  for S in ${HOLD:-PHerc1667 PHerc0139 PHerc1299 PHerc0846A}; do
    local O=$UR/$S-v2$TAG CV=CUR_$S; local C=${!CV:-${CURROOT:-$REFUR}/$S}
    unroll $S || { log "$S unroll failed ($O/progress.log)"; continue; }
    [ -d $C/wraps ] || { log "$S: no current unroll at $C (CUR_$S=DIR): candidate numbers only"; }
    for d in $O $C; do
      [ -d $d/wraps ] || continue
      j=$(grep -iE 'join|agree' $d/logs/join.log 2>/dev/null | tail -1); w=$(tail -1 $d/wrapcheck.tsv 2>/dev/null)
      log "$S $(basename $d): join [$j] wrap-check [$w] on-papyrus [$(grep -i papyrus $d/progress.log 2>/dev/null | tail -1)]"
    done
    if [ -d $C/wraps ]; then
      $U wrap-match $O/wraps $C/wraps --axis $O/axis.json > $O/wrap-match.txt 2>&1
      log "$S wrap-match: $(tail -2 $O/wrap-match.txt | tr '\n' ' ')"
      for d in $O $C; do for r in $d/img $d/strip; do [ -d $r ] && { $U ink-cover $r > $d/ink-cover-v2$TAG.txt 2>&1; break; }; done; done   # (per-wrap renders: PERWRAP=1, else the strip stripes)
      log "$S ink-cover cand: $(tail -1 $O/ink-cover-v2$TAG.txt 2>/dev/null) cur: $(tail -1 $C/ink-cover-v2$TAG.txt 2>/dev/null)"
    fi
  done
}
case $MODE in
  screen) screen ;;
  milestone) p4; hold ;;
  p4) p4 ;;
  hold) hold ;;
  *) echo "eval.sh screen|milestone|p4|hold CKPT TAG"; exit 2 ;;
esac

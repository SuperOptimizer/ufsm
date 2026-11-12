#!/usr/bin/env bash
# scripts/univ/deskgate.sh CKPT TAG PART: one gated U1 eval on the desk inside a GPU window (2026-11-12: the pod is gone; u1 trains on both
# desk cards and stops for ~45 min windows, announced by univA). Jobs run on both GPUs at once from one queue (longest first; each worker
# takes the next job when free), rows split into box quarters so the two cards finish together. PART:
#   w1   the two-phase equality test (EQ, first), winding native (DLV 0) and fed 19.2 um (DLV +1), rv p1667
#   w2   winding fed 4.8 um (DLV -1), the 9.36 um rows, xres DLV 0 / -1
#   base desk baseline rows the matrix pairs against that the desk lacks (BASES: "CKPT=TAG ..." with the DLV / xres rows to add)
# Same pinned predict build as the pod gates (UP: master 6b7fba0, built on the desk), the desk fitter ($WB/bin, fitter config 052747ed, the
# desk baselines' config), the local CT copies (LOCALCT; wbench ctpick falls back to the mirror per box). No new box starts after STOPAT
# (window start + WINMIN - 3 min); finished boxes are kept, so the next window continues where this one stopped. Then the matrix
# (eval.sh TASKS=none: file-driven, no GPU) against the desk baselines (REFS). Holds $LB/gate.busy while it runs (u1 must not restart before
# it is gone: both cards are in use).
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}; PART=${3:?w1|w2|base}
H=$(cd "$(dirname "$0")/../.." && pwd); WB=/vesuvius/ufsm/wbench; LB=${LB:-/vesuvius/ufsm/v2/lat/latres}
UP=${UP:-/home/forrest/ufsm-eval6b7/build/ufsm}; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
P4K=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr; WINMIN=${WINMIN:-45}; GPUS=${GPUS:-0 1}
Q=$LB/gate-$T-$PART.q; MC=""; [ -x ~/bin/memcap ] && MC="$HOME/bin/memcap -m 24G"; LG=$LB/gate-$T.log; EQB=${EQB:-139-1 139-2 814-1 814-2}
log() { echo "$(date '+%m-%d %H:%M:%S') [$PART] $*" | tee -a $LG; }
CORE=$(grep -v '^#' $WB/boxes.tsv | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1)
N936="139-1 139-2 139-3 139-4 139-5 139-6 139-7 139-8 814-1 814-2 814-3 814-4 814-5 814-6 814-7 814-8 500P2-1"
XRES=$(grep -v '^#' $WB/xres/boxes-xres.tsv | cut -f1)
quarters() { echo "$1" | tr ' ' '\n' | grep . | awk -v k=${2:-4} '{q[(NR - 1) % k] = q[(NR - 1) % k] " " $1} END {for (i = 0; i < k; i++) if (q[i] != "") print q[i]}'; }
# a job line: NAME|ENV...|KIND   (KIND wb: wbench.sh CK TAGROW; rv: eval.sh TASKS=rv; eq: the equality test)
WBE="SITE=desk WB=$WB SLABS=0 THREADS=12 PRIO=eval UP=$UP WZARGS= WIN=640 HALO=64 LOCALCT=/vesuvius/ufsm/v2/ct"
: > $Q
addrow() { local row=$1 boxes=$2 env=$3 k=${4:-4} i=0; while read -r b; do i=$((i + 1)); echo "$row.q$i|$env BOXES='$b'|wb $CK $row" >> $Q; done < <(quarters "$boxes" $k); }
case $PART in
  w1) echo "eq|EQ=1|eq" >> $Q; echo "rv|RVSC=p1667|rv" >> $Q
      addrow $T "$CORE" ""; addrow $T-dlv1 "$CORE" "DLV=1";;
  w2) addrow $T-dlv-1 "$CORE" "DLV=-1 P4ROOT=$M P4V=$P4K" 6; addrow $T-n936 "$N936" "" 2
      addrow $T-xres0 "$XRES" "BT=$WB/xres/boxes-xres.tsv DLV=0" 2; addrow $T-xres-1 "$XRES" "BT=$WB/xres/boxes-xres.tsv DLV=-1" 3;;
  base) for kv in ${BASES:?BASES=\"CKPT=TAG[:vox][@row,row] ...\"}; do c=${kv%%=*}; b=${kv#*=}; v=""   # :vox = a voxel net (z windows 384 / 64, not cubic)
        br=${BROWS:-dlv1 dlv-1}; [ "${b#*@}" != "$b" ] && { br=$(echo ${b#*@} | tr , " "); b=${b%%@*}; }   # @rows: this entry's rows
        [ "${b%:vox}" != "$b" ] && { b=${b%:vox}; v="WZARGS='--window-z 384 --halo-z 64' "; }
        for r in $br; do case $r in
          dlv1) CK=$c addrow $b-dlv1 "$CORE" "${v}DLV=1";; dlv-1) CK=$c addrow $b-dlv-1 "$CORE" "${v}DLV=-1 P4ROOT=$M P4V=$P4K" 6;;
          native) CK=$c addrow $b "$CORE" "$v";; n936) CK=$c addrow $b-n936 "$N936" "$v" 2;;
          xres0) CK=$c addrow $b-xres0 "$XRES" "${v}BT=$WB/xres/boxes-xres.tsv DLV=0" 2;; xres-1) CK=$c addrow $b-xres-1 "$XRES" "${v}BT=$WB/xres/boxes-xres.tsv DLV=-1" 3;;
        esac; done; done;;
  *) echo "deskgate: PART w1 | w2 | base"; exit 2;;
esac
[ "${DRY:-0}" = 1 ] && { cat $Q; rm -f $Q; exit 0; }   # DRY=1: the job list only
# the window: both cards free of the trainer (univA stops u1 for it)
TRP=${TRP:-ufsm[^ ]* train .*sources-(u1|unified)}   # the desk trainer whose stop opens the window (u1, then the unified run)
until ! pgrep -f "$TRP" > /dev/null; do sleep 20; done
T0=$(date +%s); STOPAT=$((T0 + (WINMIN - 3) * 60)); export STOPAT
# GPU tests sharing the window's first minutes (the lead's merge batches, build-verify suite binaries): my predicts start when they are
# done, at most TESTWAIT s (360) into the window
while pgrep -f 'build-verify/test_|run_suite.sh' > /dev/null && [ $(( $(date +%s) - T0 )) -lt ${TESTWAIT:-360} ]; do sleep 10; done
# GO: a file the coordinator touches when the window's GPU tests are done (w1 at 16000); the gate starts on it (or the tests ending, or TESTWAIT)
[ -n "${GO:-}" ] && while [ ! -e $GO ] && pgrep -f 'build-verify/test_|run_suite.sh' > /dev/null && [ $(( $(date +%s) - T0 )) -lt ${TESTWAIT:-360} ]; do sleep 10; done
echo "$T $PART pid $$ from $(date '+%H:%M') to $(date -d @$STOPAT '+%H:%M')" > $LB/gate.busy
log "window: $(wc -l < $Q) jobs on GPUs [$GPUS], no new box after $(date -d @$STOPAT '+%H:%M')"
eqtest() {   # the two-phase equality test: the same boxes one-phase (eq1) and predict-then-fit (eq2), the same GPU, rows equal but timings
  local g=$1 a=u1-10000-eq1 b=u1-10000-eq2 e="$WBE GPU=$g BOXES='$EQB'"
  bash -c "env $e bash $H/scripts/wbench/wbench.sh $CK0 $a" >> $LB/gate-eq.out 2>&1
  bash -c "env $e PHASE=predict bash $H/scripts/wbench/wbench.sh $CK0 $b && env $e bash $H/scripts/wbench/wbench.sh $CK0 $b" >> $LB/gate-eq.out 2>&1
  local d; d=$(diff <(cut -f2-24,29- $WB/res/$a.tsv | sort) <(cut -f2-24,29- $WB/res/$b.tsv | sort) | grep -c '^[<>]')
  log "eq: $(($(wc -l < $WB/res/$a.tsv) - 1)) / $(($(wc -l < $WB/res/$b.tsv) - 1)) rows, $d differing lines (metrics + counts; pred_s / fit_s / bin / fcfg excluded)"; }
CK0=${EQCK:-/vesuvius/ufsm/v2/lat/runs/u1/snap/10000.ckpt}
worker() {   # worker GPU: take jobs until the queue is empty or the window ends
  local g=$1 line nm env kind
  while :; do [ $(date +%s) -ge $STOPAT ] && { log "gpu $g: window end"; return; }
    line=$(flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"); [ -n "$line" ] || return
    IFS='|' read -r nm env kind <<< "$line"; log "gpu $g: $nm start"; local t0=$(date +%s)
    case $kind in
      eq) eqtest $g;;
      rv) bash -c "env GPU=$g RVU=$UP WZARGS= WIN=640 $env TASKS=rv REFS= bash $H/scripts/univ/eval.sh $CK $T" >> $LB/gate-$T-rv.out 2>&1;;
      wb*) set -- $kind; bash -c "env $WBE GPU=$g $env $MC bash $H/scripts/wbench/wbench.sh $2 $3" >> $LB/gate-$T-$nm.out 2>&1;;
    esac; log "gpu $g: $nm done rc=$? in $(( $(date +%s) - t0 )) s"
  done; }
for g in $GPUS; do worker $g & done; wait
left=$(wc -l < $Q); log "window done: $left jobs not started$( [ $left -gt 0 ] && echo ': '$(cut -d'|' -f1 $Q | tr '\n' ' '))"
rm -f $LB/gate.busy
[ $PART = base ] || { TASKS=none REFS="${REFS:-wind96 lat-wz896-12000}" bash $H/scripts/univ/eval.sh $CK $T > /dev/null 2>&1; log "matrix -> $WB/univ/$T.matrix.txt"; }

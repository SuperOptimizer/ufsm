#!/bin/bash
# scripts/pod/deskq2.sh: podq2's unroll on the desk (forlindesk2) instead of a pod (user 11-18: no pods). Full PHerc1667 at the master
# commit SNAP was made from, u1 13790, every speedup (r3 + ink-once, m7 prefetch, render alias, the fields' fused head / fstem: master
# defaults), PERWRAP 1, KEEPINK 1, CLEAN 0, R3BATCH 16 (8 on a reader3 out-of-memory, resumed). Outputs straight to OUT (no sync step).
#   1. waits for the card-1 chain's markers (fix0332 gn1667, then pseudolab res48: WAITFOR) IN THE SCRIPT, not on a message
#   2. card 1 only: CUDA_VISIBLE_DEVICES=1 for every stage, all GPU vars 0, SGPU unset (each stage inherits the one card); a watcher checks
#      every 20 s that no process of this run holds memory on another card (nvidia-smi gpu_uuid) and stops the run (STOP) if one does
#   3. flock -s the suite's gpu1.excl, then opt/gpu1.lock (the team rule); memcap dev slice + lowprio (caller), 8 threads
#   4. at the end: DONE (q2.done, written success or failure, with rc), then scripts/pod/score1667.sh OUT and OUT/stagetable.tsv
# Desk timings (5060 Ti 16 GB, a card shared with try9's 1.8-3.7 GB) are not pod timings.
# Usage (detached): SNAP=<dir with bin/ufsm + scripts/> setsid nohup ~/bin/memcap ~/bin/lowprio bash $SNAP/scripts/pod/deskq2.sh > LOG 2>&1 < /dev/null &
set -u
SNAP=${SNAP:?SNAP: the snapshot dir (bin/ufsm, scripts/, COMMIT)}; U=$SNAP/bin/ufsm; R=$SNAP/scripts
OUT=${OUT:-/vesuvius/ufsm/podun/PHerc1667-q2}; DONE=${DONE:-/vesuvius/ufsm/unrollperf/q2b.done}   # (q2.done: the first, card-0-faulted run, 11-18 08:28)
WAITFOR=${WAITFOR:-/vesuvius/ufsm/pseudolab/res48/halo.done}   # (the rerun after the 11-18 card-0 fault: pseudolab's halo check, chained on q2.done)
CK=${CK:-/vesuvius/ufsm/v2/lat/runs/u1-donor-13790.ckpt}; CKSHA1=${CKSHA1:-8854e4e5bd8ede2133c70aada6032a5b07715bb0}
say() { echo "$(date '+%F %T') deskq2: $*"; }
fin() { echo "$(date '+%F %T') rc $1 $(cat $SNAP/COMMIT) $OUT" > $DONE.part && mv -f $DONE.part $DONE; say "wrote $DONE (rc $1)"; }
[ "$(sha1sum $CK | cut -d' ' -f1)" = $CKSHA1 ] || { say "checkpoint $CK: sha1 mismatch"; exit 2; }
case "$OUT" in /vesuvius/ufsm/podun/PHerc1667-*) ;; *) say "OUT must be /vesuvius/ufsm/podun/PHerc1667-*"; exit 2;; esac
say "start: commit $(cat $SNAP/COMMIT), waiting for: $WAITFOR"
for m in $WAITFOR; do w=0; while [ ! -e $m ]; do [ $w = 0 ] && say "waiting for $m"; w=1; sleep 30; done; say "$m present: $(head -c 200 $m 2>/dev/null)"; done
f=$(df --output=avail -BG /vesuvius | tail -1 | tr -dc 0-9); [ "$f" -ge 100 ] || { say "only $f GB free on /vesuvius"; fin 2; exit 1; }
export CUDA_VISIBLE_DEVICES=1 FIELDGPUS=0 FITGPUS=0 INKGPU=0; unset SGPU
export UR=$(dirname $OUT) TAG=${OUT#$(dirname $OUT)/PHerc1667} U CK PERWRAP=1 KEEPINK=1 CLEAN=0 INKMODE=r3 R3ONCE=1 R3BATCH=${R3BATCH:-16} SPIRALDIR=-1 \
       TH=8 ITH=8 PTH=8 R3TH=8 SPAR=2 CFMEM=${CFMEM:-20} UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4} UFSM_SAMPLER_THP=0
C0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 0 {print $2}')
guard() {   # the run's process tree must not hold memory on card 0 (the training card): every 5 s; on a hit STOP and KILL the whole
  # tree at once (11-18: the first guard only touched STOP, which stops new jobs; the card-0 flatten / render / reader3 ran on for 75 s)
  while kill -0 $1 2> /dev/null; do
    local tree hit; tree=" $(pstree -p $1 2>/dev/null | grep -oE '\([0-9]+\)' | tr -d '()' | tr '\n' ' ') "
    hit=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r p g; do
      [ "$g" = "$C0" ] && case "$tree" in *" $p "*) echo "$p";; esac; done)
    if [ -n "$hit" ]; then touch $OUT/STOP; say "GUARD: pid(s) $hit of the run on card 0 ($(ps -o args= -p ${hit%% *} | cut -c1-120)): STOP + kill the run"
      kill -KILL $tree 2> /dev/null; touch $OUT/.guard_killed; return 1; fi
    sleep 5; done; }
run() { flock -s /tmp/ufsm-suite-locks/gpu1.excl flock -s /vesuvius/ufsm/opt/gpu1.lock bash $R/unroll/unroll.sh PHerc1667 & local p=$!; guard $p & local g=$!; wait $p; local rc=$?; kill $g 2> /dev/null; return $rc; }
# preflight (11-18): the card placement test, with one real CUDA call on the card pcard picks (gpu_uuid must be card 1's)
C1=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 1 {sub("GPU-", "", $2); print $2}')
( cd $SNAP && CARD=1 CARD_UUID=$C1 U=$U flock -s /tmp/ufsm-suite-locks/gpu1.excl bash $SNAP/tests/test_unroll_cards.sh ) > $OUT.preflight.log 2>&1 \
  || { say "preflight FAILED ($OUT.preflight.log)"; fin 3; exit 3; }
say "preflight ok: $(grep 'real GPU call' $OUT.preflight.log)"
mkdir -p $OUT; rm -f $OUT/STOP $OUT/.guard_killed; say "unroll -> $OUT (R3BATCH $R3BATCH)"; run; rc=$?
if [ $rc != 0 ] && [ ! -e $OUT/STOP ] && [ ! -e $OUT/.guard_killed ] && grep -lqiE "out of memory" $OUT/logs/r3*.log $OUT/r/*/render.log 2> /dev/null && [ $R3BATCH -gt 8 ]; then
  say "reader3 out of memory at R3BATCH $R3BATCH: resuming at 8"; export R3BATCH=8; run; rc=$?; fi
[ $rc = 0 ] && [ -e $OUT/state/stripimg.done ] && [ -e $OUT/state/perwrap.done ] || { [ $rc = 0 ] && rc=9; }
awk '/ start$/ {s[$3] = $1 " " $2} / done: / {st = $3; sub(/:$/, "", st); printf "%s\t%s\t%s %s\t%s\t%s\t%s\n", st, s[st], $1, $2, $5, $8, $11}' $OUT/progress.log \
  | awk -F'\t' 'BEGIN {print "stage\tstart\tend\twall_s\tcpu_s\tmax_rss_gb"} {print}' > $OUT/stagetable.tsv
fin $rc; touch $OUT/.synced
[ $rc = 0 ] && { say "scoring"; CUDA_VISIBLE_DEVICES= U=$U bash $R/pod/score1667.sh $OUT > $OUT/score.out 2>&1; say "score: $(cat $OUT/score/score.txt 2>/dev/null | tr '\n' ' ')"; }
say "end rc $rc"; exit $rc

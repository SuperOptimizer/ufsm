#!/usr/bin/env bash
# scripts/universal/launch.sh RUN_DIR [extra dino-train options]: the universal model (dino-train --recipe 1, v0 config) on desk GPU 0 under a
# watchdog: setsid it (launch.sh starts itself detached unless FG=1); each (re)start rebuilds the sources (new U0 scans join) and resumes
# from RUN_DIR/last.ckpt; RUN_DIR/STOP stops it (the trainer saves and exits; the watchdog does not restart). Checkpoints every SAVE steps,
# milestones every KEEP kept. US: the unlabelled share (0: labelled only). Weights: GT / OBJ / REG (--gt / --object / --region, default 0.1); DENSE=1 adds the dense heads (UW / IKP weights). Log: RUN_DIR/train.log (steps), RUN_DIR/stderr.log, RUN_DIR/watchdog.log.
set -u
R=${1:?run dir}; shift; mkdir -p $R; R=$(cd $R && pwd)
if [ "${FG:-0}" != 1 ]; then FG=1 setsid nohup bash "$0" "$R" "$@" > $R/watchdog.log 2>&1 < /dev/null & echo "watchdog started (pid $!), log $R/watchdog.log"; exit 0; fi
D=$(cd "$(dirname "$0")" && pwd); U=${U:-/vesuvius/ufsm/dino/src/build/ufsm}; GPU=${GPU:-0}
STEPS=${STEPS:-1000000}; SAVE=${SAVE:-2000}; KEEP=${KEEP:-20000}
P=${P:-256}; SP=${SP:-384}; LP=${LP:-96}; ZOOM=${ZOOM:-0.7,1.4}
CFG="--recipe 1 --P $P --source-P $SP --local-P $LP --zoom $ZOOM --attn-blocks 2 --protos-k 32768 --ibot-k 32768 --linear-mx 4 --head-slab 16 --mem lean
     --unlabelled-share ${US:-0.6} --exclude-src PHerc1667 --gt ${GT:-0.1} --object ${OBJ:-0.1} --region ${REG:-0.1} --dim 32 --ssl 0 --warmup 2000 --teacher-temp-warmup 30000 --freeze-last 1250"
# DENSE=1: the dense heads (unwrap2 + ink-prob, robust phase loss; a 32-output checkpoint grows to 48 on resume, the heads ramp in over 2000 steps)
[ "${DENSE:-0}" = 1 ] && CFG="$CFG --unwrap2 ${UW:-1} --ink-prob ${IKP:-1} --wind4-lambda 0.5 --dense-ramp 2000 --dense-from ${DFROM:?DENSE=1 needs DFROM (the step the dense heads start at: fixed across watchdog restarts)} --phase-trunc 0.25 --phase-trunc-conf 0.5 --normal-step 1 --density-scale 4 --density-code 1 --packed-weight 8 --packed-um 120"
fast=0
for try in $(seq 1 1000); do
  [ -e $R/STOP ] && { echo "$(date) STOP"; break; }
  [ -e $R/done ] && { echo "$(date) done"; break; }
  bash $D/mksources.sh $R/sources.json >> $R/watchdog.log 2>&1 || { echo "$(date) mksources failed"; sleep 300; continue; }
  RES=""; [ -e $R/last.ckpt ] && RES="--resume $(readlink -f $R/last.ckpt)"
  echo "$(date) start $try $RES"; t0=$(date +%s)
  $U dino-train --gpu $GPU --sources $R/sources.json --out $R --steps $STEPS --save-every $SAVE --keep-every $KEEP --log-every 100 --eval-every 100000000 \
     --workers ${WORKERS:-12} --slots ${SLOTS:-12} --threads 4 $CFG $RES "$@" >> $R/stderr.log 2>&1; rc=$?
  echo "$(date) exit $rc"
  if [ $(( $(date +%s) - t0 )) -lt 300 ] && [ $rc != 0 ]; then fast=$((fast + 1)); else fast=0; fi
  if [ $fast -ge 5 ]; then echo "$(date) !!! 5 failed starts in a row (each < 5 min): watchdog stops; see $R/stderr.log"; touch $R/FAILED; break; fi
  [ $rc = 0 ] && [ ! -e $R/STOP ] && touch $R/done   # (a STOP exit is not the end: remove STOP and relaunch to continue)
  sleep 60
done

#!/usr/bin/env bash
# scripts/universal/launch.sh RUN_DIR [extra dino-train options]: the universal model (dino-train --recipe 1, v0 config) on desk GPU 0 under a
# watchdog: setsid it (launch.sh starts itself detached unless FG=1); each (re)start rebuilds the sources (new U0 scans join) and resumes
# from RUN_DIR/last.ckpt; RUN_DIR/STOP stops it (the trainer saves and exits; the watchdog does not restart). Checkpoints every SAVE steps,
# milestones every KEEP kept. Log: RUN_DIR/train.log (steps), RUN_DIR/stderr.log, RUN_DIR/watchdog.log.
set -u
R=${1:?run dir}; shift; mkdir -p $R; R=$(cd $R && pwd)
if [ "${FG:-0}" != 1 ]; then FG=1 setsid nohup bash "$0" "$R" "$@" > $R/watchdog.log 2>&1 < /dev/null & echo "watchdog started (pid $!), log $R/watchdog.log"; exit 0; fi
D=$(cd "$(dirname "$0")" && pwd); U=${U:-/vesuvius/ufsm/dino/src/build/ufsm}; GPU=${GPU:-0}
STEPS=${STEPS:-1000000}; SAVE=${SAVE:-2000}; KEEP=${KEEP:-20000}
P=${P:-256}; SP=${SP:-384}; LP=${LP:-96}; ZOOM=${ZOOM:-0.7,1.4}
CFG="--recipe 1 --P $P --source-P $SP --local-P $LP --zoom $ZOOM --attn-blocks 2 --protos-k 32768 --ibot-k 32768 --linear-mx 4 --head-slab 16 --mem lean
     --unlabelled-share 0.6 --exclude-src PHerc1667 --gt 0.1 --dim 32 --ssl 0 --warmup 2000 --teacher-temp-warmup 30000 --freeze-last 1250"
for try in $(seq 1 1000); do
  [ -e $R/STOP ] && { echo "$(date) STOP"; break; }
  [ -e $R/done ] && { echo "$(date) done"; break; }
  bash $D/mksources.sh $R/sources.json >> $R/watchdog.log 2>&1 || { echo "$(date) mksources failed"; sleep 300; continue; }
  RES=""; [ -e $R/last.ckpt ] && RES="--resume $(readlink -f $R/last.ckpt)"
  echo "$(date) start $try $RES"
  $U dino-train --gpu $GPU --sources $R/sources.json --out $R --steps $STEPS --save-every $SAVE --keep-every $KEEP --log-every 100 --eval-every 100000000 \
     --workers 6 --threads 6 $CFG $RES "$@" >> $R/stderr.log 2>&1; rc=$?
  echo "$(date) exit $rc"
  [ $rc = 0 ] && [ ! -e $R/STOP ] && touch $R/done   # (a STOP exit is not the end: remove STOP and relaunch to continue)
  sleep 60
done

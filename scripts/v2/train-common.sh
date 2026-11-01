#!/usr/bin/env bash
# v2 universal model (plan bright-forging-seahorse, Phase 1): shared launcher body of train-a.sh / train-b.sh. Not run directly.
# Task unwrap2 + --ink-head 1 (phase, n n^T, density, winding mod 4, ink logit = channel 11), sources from scripts/v2/mksources.sh
# (per source: targets.band.phase + ignore, targets.ink {"root", "group", "encoding": "prob254"}, "weight", "ink_weight").
# Desk only (2x RTX 5060 Ti 16 GB), writes under /vesuvius/ufsm/v2 only. Auto-resume from <run>/last.ckpt, STOP file <run>.stop
# (checked between relaunches; a running trainer is stopped by its own PID: kill -TERM $(cat <run>.pid), graceful, rc 0).
# Env: U (binary), SRCJ (sources), P (512), WORKERS (12), SLOTS (16), MEM (lean), CKPT (1000), VAL (2000), VALB (16), LOGE (50), EXTRA.
set -u
V=/vesuvius/ufsm/v2
U=${U:-$V/bin/ufsm}; SRCJ=${SRCJ:-$V/sources/sources-v2.json}
[ -n "${NAME:-}" ] && [ -n "${STEPS:-}" ] && [ -n "${GPU:-}" ] && [ -n "${ARCH:-}" ] || { echo "train-common.sh: set NAME STEPS GPU ARCH (use train-a.sh / train-b.sh)"; exit 2; }
o=$V/runs/$NAME; mkdir -p $V/runs; cd $V || exit 1
export MALLOC_ARENA_MAX=4
[ -x "$U" ] || { echo "no binary $U"; exit 1; }
[ -f "$SRCJ" ] || { echo "no sources $SRCJ (scripts/v2/mksources.sh)"; exit 1; }
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$avail" -ge 40 ] || { echo "MemAvailable ${avail} GB < 40 GB: not starting"; exit 1; }
# augmentation: wind96's recipe (z-fixed symmetries at p 0.5, 5 degree rotations, elastic, the scanner domains), axis jitter
AUG="--zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20 \
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0 \
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-hist-p 0.15 --aug-hist /vesuvius/ufsm/slab/target-hist.txt --aug-noise-p 0.15 --aug-cnoise-p 0.1 \
  --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1"
# heads: wind96's density code / packed weight / normal step; ink soft BCE
HEADS="--task unwrap2 --ink-head 1 --ink-lambda ${INKL:-1} --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1"
ARGS=($SRCJ --out $o --gpus $GPU --P ${P:-512} --B 1 $ARCH $HEADS --steps $STEPS --sched wsd --cooldown 0.2 --opt muon \
  --fp4 2 --gpu-aug 2 --input-prec 8 --mem ${MEM:-lean} --val-store disk --workers ${WORKERS:-12} --sampler-slots ${SLOTS:-16} \
  --log-every ${LOGE:-50} --val-every ${VAL:-2000} --val-batches ${VALB:-16} --ckpt-every ${CKPT:-1000} --seed ${SEED:-3} --val-seed 1 \
  $LRARGS $AUG ${EXTRA:-})
echo $$ > $o.launcher.pid
for i in $(seq 0 ${RELAUNCH:-8}); do
  [ -f $o.stop ] && { echo "$(date) STOP file $o.stop" >> $o.resume.log; break; }
  R=(); [ -f $o/last.ckpt ] && R=(--resume $o/last.ckpt) || R=($INIT)
  echo "$(date) start $i: $U train ${ARGS[*]} ${R[*]}" >> $o.resume.log
  nice -n 5 $U train "${ARGS[@]}" "${R[@]}" >> $o.log 2>&1 & echo $! > $o.pid; wait $!; rc=$?
  echo "$(date) exit rc=$rc" >> $o.resume.log
  last=$(awk '/^step +[0-9]+ lr/ {s=$2} END {print s}' $o.log)
  [ "${last:-0}" -ge "$STEPS" ] && break
  [ $rc = 0 ] && [ -f $o.stop ] && break
  sleep 60
done
rm -f $o.pid
echo "$(date) done" >> $o.resume.log

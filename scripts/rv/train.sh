#!/usr/bin/env bash
# scripts/rv/train.sh NAME STEPS: the recto / verso side model, FROM SCRATCH (no --init-from), desk or pod.
# train --task unwrap2 --rv 1: channels 0-1 = recto-side / verso-side papyrus (the user's rule: between consecutive wraps k and k + 1,
# the half nearer k is recto; papyrus = CT >= the source's "iso"), plus the unwrap2 geometry heads from the same phase (normal n n^T,
# density, winding mod 4 at WIND4L) as auxiliary losses. 7 levels 16..256 (the large receptive field for packed sheets), MX-fp4,
# --mem auto (the planner's leanest storage that fits), --gpu-aug 2, packed sheets x8 (also on the rv loss).
# Sources: scripts/rv/mksources.sh. Auto-resume from <run>/last.ckpt; STOP file <run>.stop (checked between relaunches; a running
# trainer: kill -TERM $(cat <run>.pid), graceful). Checkpoints every CKPT (1000) steps, snapshots kept every SNAP (5000).
# Env: R (run root, /vesuvius/ufsm/v2/rv/runs), U (binary), SRCJ, GPU (0), LOCK (flock file; "" = none), P (576), ARCH, WORKERS (12),
#      SLOTS (16), MEMMODE (auto), ALR / MLR, WIND4L (0.25), ISO (64), EXTRA. Needs MemAvailable >= MINMEM GB (40).
set -u
NAME=${1:?NAME}; STEPS=${2:?STEPS}
R=${R:-/vesuvius/ufsm/v2/rv/runs}; U=${U:-/vesuvius/ufsm/v2/rv/bin/ufsm}; SRCJ=${SRCJ:-/vesuvius/ufsm/v2/rv/sources-rv.json}
GPU=${GPU:-0}; LOCK=${LOCK-/vesuvius/ufsm/opt/gpu$GPU.lock}; P=${P:-576}
ARCH=${ARCH:---widths 16,16,32,64,128,192,256}
o=$R/$NAME; mkdir -p $o; cd $R || exit 1
export MALLOC_ARENA_MAX=4
[ -x "$U" ] || { echo "no binary $U"; exit 1; }
[ -f "$SRCJ" ] || { echo "no sources $SRCJ"; exit 1; }
cp -n $SRCJ $o/sources.json.first 2>/dev/null; cp $U $o/ufsm.bin 2>/dev/null
AUG="--zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20 \
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0 \
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-noise-p 0.15 --aug-cnoise-p 0.1 \
  --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1 ${HISTAUG:-}"
HEADS="--task unwrap2 --rv 1 --rv-iso ${ISO:-64} --wind4-lambda ${WIND4L:-0.25} --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1"
ARGS=($SRCJ --out $o --gpus 0 --P $P --B 1 $ARCH $HEADS --steps $STEPS --sched wsd --cooldown 0.2 --opt muon \
  --fp4 2 --gpu-aug 2 --input-prec 8 --mem ${MEMMODE:-auto} --val-store disk --workers ${WORKERS:-12} --sampler-slots ${SLOTS:-16} \
  --log-every ${LOGE:-50} --val-every ${VAL:-2000} --val-batches ${VALB:-16} --ckpt-every ${CKPT:-1000} --seed ${SEED:-3} --val-seed 1 \
  --warmup 2000 --muon-lr ${MLR:-0.02} --lr ${ALR:-0.001} $AUG ${EXTRA:-})
echo $$ > $o.launcher.pid
# snapshots: every SNAP steps last.ckpt is copied to <run>/snap/<step>.ckpt (the eval checkpoints); the copy's own header names its step
mkdir -p $o/snap
( while kill -0 $$ 2>/dev/null; do
    if [ -f $o/last.ckpt ] && [ $o/last.ckpt -nt $o/snap/.seen ]; then
      sleep 20; cp $o/last.ckpt $o/snap/.tmp && touch $o/snap/.seen
      st=$(head -c 400 $o/snap/.tmp | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2)
      if [ -n "$st" ] && [ $((st % ${SNAP:-5000})) = 0 ] && [ "$st" -gt 0 ]; then mv $o/snap/.tmp $o/snap/$st.ckpt; echo "$(date) snapshot $st" >> $o.resume.log; fi
    fi; sleep 60; done ) &
for i in $(seq 0 ${RELAUNCH:-20}); do
  [ -f $o.stop ] && { echo "$(date) STOP file $o.stop" >> $o.resume.log; break; }
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt ${MINMEM:-40} ]; then echo "$(date) MemAvailable ${avail} GB < ${MINMEM:-40}: waiting" >> $o.resume.log; sleep 300; continue; fi
  RS=(); [ -f $o/last.ckpt ] && RS=(--resume $o/last.ckpt)
  echo "$(date) start $i (GPU $GPU, lock ${LOCK:-none}): $U train ${ARGS[*]} ${RS[*]}" >> $o.resume.log
  L=(); [ -n "$LOCK" ] && L=(flock $LOCK)
  CUDA_VISIBLE_DEVICES=$GPU "${L[@]}" nice -n 5 $U train "${ARGS[@]}" "${RS[@]}" >> $o.log 2>&1 & echo $! > $o.pid; wait $!; rc=$?
  echo "$(date) exit rc=$rc" >> $o.resume.log
  last=$(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $o.log)
  [ "$last" -ge "$STEPS" ] && break
  [ $rc = 0 ] && [ -f $o.stop ] && break
  sleep 60
done
rm -f $o.pid
echo "$(date) done" >> $o.resume.log

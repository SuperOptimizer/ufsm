#!/usr/bin/env bash
# scratch2.train.sh (merger2 2026-11-14, ops audit #5 / #6): u1.train2.sh for the scratch2 run, with three changes:
#  (1) <run>.relaunch = "bash s2ctl.sh resume" (launch.env is the only config; the old line dropped UFSM_HEAD_SLAB_Z / UFSM_HOSTMEM / WARM / SPLITF / ONFLYSLOTS);
#  (2) every trainer start appends its UFSM_* / MALLOC_* / CUDA_* environment and argv to <run>.resume.log;
#  (3) a stop file found at start is logged with its owner / reason (and printed) instead of a silent exit.
# (u1.train2.sh, 2026-11-11: the relaunch loop runs outside the memcap scope, only the trainer inside it, so an oomd kill of the trainer is relaunched from last.ckpt; checkpoints every 250 steps)
# u1.train.sh (a copy of scripts/rv/train.sh for the universal run u1: --rv 2, warmup 300, ckpt 500, no validation batches, snapshots every 3000 as defaults)
# scripts/rv/train.sh NAME STEPS: the recto / verso side model, FROM SCRATCH (no --init-from), desk or pod.
# train --task unwrap2 --rv 1: channels 0-1 = recto-side / verso-side papyrus (the user's rule: between consecutive wraps k and k + 1,
# the half nearer k is recto; papyrus = CT >= the source's "iso"), plus the unwrap2 geometry heads from the same phase (normal n n^T,
# density, winding mod 4 at WIND4L) as auxiliary losses. 7 levels 16..256 (the large receptive field for packed sheets), MX-fp4,
# --mem auto (the planner's leanest storage that fits), --gpu-aug 2, packed sheets x8 (also on the rv loss).
# Sources: scripts/rv/mksources.sh. Auto-resume from <run>/last.ckpt; STOP file <run>.stop (checked between relaunches; a running
# trainer: kill -TERM $(cat <run>.pid), graceful). Checkpoints every CKPT (1000) steps, snapshots kept every SNAP (5000).
# Env: R (run root, /vesuvius/ufsm/v2/rv/runs), U (binary), SRCJ, GPU (0, or "0,1": both, data parallel), LOCK (flock files, comma list; "" = none), P (576), ARCH, WORKERS (12),
#      SLOTS (16), MEMMODE (auto), ALR / MLR, WIND4L (0.25), ISO (64), EXTRA. Needs MemAvailable >= MINMEM GB (40).
#      MAXFAIL (3): the same nonzero rc that many times in a row with no new step stops the relaunch loop.
set -u
NAME=${1:?NAME}; STEPS=${2:?STEPS}
R=${R:-/vesuvius/ufsm/v2/rv/runs}; U=${U:-/vesuvius/ufsm/v2/rv/bin/ufsm}; SRCJ=${SRCJ:-/vesuvius/ufsm/v2/rv/sources-rv.json}
GPU=${GPU:-0}; P=${P:-576}   # GPU: one id or a list "0,1" (data parallel over them: --gpus 0..n-1 of CUDA_VISIBLE_DEVICES=GPU)
LOCK=${LOCK-$(echo $GPU | tr , '\n' | sed 's|.*|/vesuvius/ufsm/opt/gpu&.lock|' | paste -sd,)}   # comma list of flock files, one per GPU
NG=$(echo $GPU | tr , '\n' | wc -l); GPUS=$(seq -s, 0 $((NG - 1)))
ARCH=${ARCH:---widths 16,16,32,64,128,192,256}
o=$R/$NAME; mkdir -p $o; cd $R || exit 1
if [ "${DRY:-0}" = 1 ]; then   # (s2ctl.sh drystart: the exact launch for 2 windows from a copy of scratch2/last.ckpt, in <R>/<NAME> = scratch2.dry)
  if [ "${FRESH:-0}" = 1 ]; then rm -f $o/last.ckpt; st=0   # (s2ctl.sh fresh: a from-scratch start, 2 windows, no checkpoint)
  else cp $R/scratch2/last.ckpt $o/last.ckpt || exit 1; st=$(head -c 200000 $o/last.ckpt | head -1 | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2); fi
  STEPS=$((${st:-0} + ${DRYSTEPS:-2})); LOGE=1; CKPT=1000000; RELAUNCH=0; SNAP=1000000000
fi
export MALLOC_ARENA_MAX=4
[ -x "$U" ] || { echo "no binary $U"; exit 1; }
[ -f "$SRCJ" ] || { echo "no sources $SRCJ"; exit 1; }
cp -n $SRCJ $o/sources.json.first 2>/dev/null; cp $U $o/ufsm.bin 2>/dev/null
AUG="--zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20 \
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0 \
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-noise-p 0.15 --aug-cnoise-p 0.1 \
  --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1 ${HISTAUG:-}"
HEADS=${HEADS:-"--task unwrap2 --rv 2 --rv-iso ${ISO:-64} --wind4-lambda ${WIND4L:-0.25} --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1"}   # (u1: --rv 2, baked in: the relaunch line does not carry HEADS)   # (HEADS: another task on the same recipe, e.g. the winding heads without --rv)
ARGS=($SRCJ --out $o --gpus $GPUS --P $P --B 1 $ARCH $HEADS --steps $STEPS --sched wsd --cooldown 0.2 --opt muon \
  --fp4 2 --gpu-aug 3 --mem ${MEMMODE:-auto} --val-store disk --workers ${WORKERS:-12} --sampler-slots ${SLOTS:-16} \
  --log-every ${LOGE:-50} --val-every ${VAL:-2000} --val-batches ${VALB:-0} --ckpt-every ${CKPT:-250} --seed ${SEED:-3} --val-seed 1 \
  --warmup ${WARM:-300} --muon-lr ${MLR:-0.02} --lr ${ALR:-0.001} $AUG ${EXTRA:-})
echo $$ > $o.launcher.pid
printf 'S2CTL_OWNER=${S2CTL_OWNER:-relaunch-file} bash %q resume\n' "$R/s2ctl.sh" > $o.relaunch   # (evwatch.sh / u1.pause.sh: bash <run>.relaunch; s2ctl reads launch.env)
# snapshots: every SNAP steps last.ckpt is copied to <run>/snap/<step>.ckpt (the eval checkpoints); the copy's own header names its step
mkdir -p $o/snap
# (ops audit #11: a binary with --snap-every keeps reflink snapshots + the --ckpt-ring itself; the shell copy loop only for older binaries)
SNAPB=0; strings "$U" | grep -qF -- "--snap-every" && SNAPB=1
[ $SNAPB = 1 ] && [ "${DRY:-0}" != 1 ] && ARGS+=(--snap-every ${SNAP:-1000} --snap-keep-every ${SNAPKEEP:-5000} --snap-keep-last ${SNAPLAST:-5})
[ "${DRY:-0}" = 1 ] || [ $SNAPB = 1 ] || ( while kill -0 $$ 2>/dev/null; do
    if [ -f $o/last.ckpt ] && [ $o/last.ckpt -nt $o/snap/.seen ]; then
      sleep 20; cp $o/last.ckpt $o/snap/.tmp && touch $o/snap/.seen
      st=$(head -c 200000 $o/snap/.tmp | head -1 | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2)
      if [ -n "$st" ] && [ $((st % ${SNAP:-3000})) = 0 ] && [ "$st" -gt 0 ]; then mv $o/snap/.tmp $o/snap/$st.ckpt; echo "$(date) snapshot $st" >> $o.resume.log; fi
    fi; sleep 60; done ) &
for i in $(seq 0 ${RELAUNCH:-20}); do
  [ -f $o.stop ] && { m="$(date) STOP file $o.stop: $(tr '\n' ' ' < $o.stop)"; echo "$m" >> $o.resume.log; [ $i = 0 ] && echo "refusing to start: $m (s2ctl.sh resume removes it)" >&2; break; }
  # (ops audit #12: the binding limit is ufsm-train.slice, not the host: memory.max - (anon + shmem) of the slice, GB)
  SL=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/ufsm.slice/ufsm-train.slice
  if [ -r $SL/memory.max ] && [ "$(cat $SL/memory.max)" != max ]; then
    avail=$(awk -v m=$(cat $SL/memory.max) '/^anon |^shmem / {u += $2} END {print int((m - u) / 1073741824)}' $SL/memory.stat)
  else avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); fi
  if [ "$avail" -lt ${MINMEM:-40} ]; then echo "$(date) MemAvailable ${avail} GB < ${MINMEM:-40}: waiting" >> $o.resume.log; sleep 300; continue; fi
  RS=(); [ -f $o/last.ckpt ] && RS=(--resume $o/last.ckpt)
  echo "$(date) start $i (GPU $GPU, lock ${LOCK:-none}): $U train ${ARGS[*]} ${RS[*]}" >> $o.resume.log
  # the GPU lock is held by this launcher (fd 9) for its lifetime, so <run>.pid is the trainer itself (SIGTERM = graceful save)
  if [ -n "$LOCK" ] && [ -z "${LOCKED:-}" ]; then fd=9; for lk in ${LOCK//,/ }; do eval "exec $fd> $lk"; flock $fd; fd=$((fd + 1)); done; LOCKED=1; echo "$(date) holding $LOCK" >> $o.resume.log; fi
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 /home/forrest/bin/memcap -t $U train "${ARGS[@]}" "${RS[@]}" >> $o.log 2>&1 & echo $! > $o.pid
  ( tp=$(cat $o.pid); sleep 15; { echo "$(date) trainer pid $tp env: $(tr '\0' '\n' < /proc/$tp/environ 2>/dev/null | grep -E '^(UFSM_|MALLOC_|CUDA_)' | sort | tr '\n' ' ')"
      echo "$(date) trainer pid $tp argv: $(tr '\0' ' ' < /proc/$tp/cmdline 2>/dev/null)"; } >> $o.resume.log ) &
  wait $(cat $o.pid); rc=$?
  echo "$(date) exit rc=$rc" >> $o.resume.log
  last=$(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $o.log)
  [ "$last" -ge "$STEPS" ] && break
  [ $rc = 0 ] && [ -f $o.stop ] && break
  # a deterministic failure (no new step, same nonzero rc: e.g. a device OOM at start) stops the loop after MAXFAIL (3) tries
  if [ $rc != 0 ] && [ "$last" = "${prev_last:-x}" ] && [ $rc = "${prev_rc:-x}" ]; then nfail=$((${nfail:-1} + 1)); else nfail=1; fi
  prev_last=$last; prev_rc=$rc
  if [ $rc != 0 ] && [ $nfail -ge ${MAXFAIL:-3} ]; then echo "$(date) rc=$rc $nfail times in a row with no new step (last $last): giving up, see $o.log" >> $o.resume.log; break; fi
  sleep 60
done
rm -f $o.pid
echo "$(date) done" >> $o.resume.log

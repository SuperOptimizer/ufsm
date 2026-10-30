#!/usr/bin/env bash
# scripts/vwt/train.sh RUN GPU [extra vwt-train args...]: one vwt (ray-attention winding model) training lane, supervised.
#   Runs `ufsm vwt-train SOURCES RUNROOT/RUN --gpu GPU --resume 1 ...` under nice; restarts it after a crash (resumes from the run's latest
#   checkpoint, at most MAXRESTART times, 60 s apart); exits when RUNROOT/RUN.stop exists (touch it, then the trainer is sent SIGTERM and
#   saves its last committed step) or when the trainer finishes (exit 0). Log: RUNROOT/RUN.log (appended), lane events RUNROOT/RUN.lane.
# Host presets (HOST=desk | forlindesk, default from hostname): binary U, sources SRC, run root RUNROOT, threads. Env overrides any of
#   U, SRC, RUNROOT, STEPS (200000), LR (5e-4), BATCH (2), EMA (0.999), TILT (15), CKPT (2000), MAXRESTART (20), NICE (5), AUG (extra aug flags).
# Build: desk `make -j16 build/ufsm`; forlindesk (2x RTX 3090, sm_86) `PATH=/usr/local/cuda-13.4/bin:$PATH make ARCH=86 CUDA=/usr/local/cuda-13.4 -j20 build86/ufsm`.
# forlindesk: keep the 225 W cap (sudo -n nvidia-smi -pl 225 after a reboot) and MemAvailable > 10 GB (the lane waits for it at each start).
set -u
[ $# -ge 2 ] || { sed -n '2,12p' "$0"; exit 2; }
RUN=$1; GPU=$2; shift 2
HOST=${HOST:-$(case $(hostname) in forlindesk2*) echo desk;; forlindesk*) echo forlindesk;; *) echo desk;; esac)}
case $HOST in
  desk)       U=${U:-/vesuvius/ufsm/vwt/src/build/ufsm};   SRC=${SRC:-/vesuvius/ufsm/vwt/sources.json}; RUNROOT=${RUNROOT:-/vesuvius/ufsm/vwt/runs} ;;
  forlindesk) U=${U:-/vesuvius/ufsm/vwt/src/build86/ufsm}; SRC=${SRC:-/vesuvius/ufsm/vwt/sources.json}; RUNROOT=${RUNROOT:-/vesuvius/ufsm/vwt/runs} ;;
  *) echo "HOST desk | forlindesk"; exit 2 ;;
esac
O=$RUNROOT/$RUN; mkdir -p "$O"; LOG=$O.log; LANE=$O.lane; STOPF=$O.stop
[ -x "$U" ] || { echo "no binary $U"; exit 2; }
[ -s "$SRC" ] || { echo "no sources $SRC"; exit 2; }
ARGS=("$SRC" "$O" --gpu "$GPU" --steps "${STEPS:-200000}" --lr "${LR:-5e-4}" --batch "${BATCH:-2}" --ema "${EMA:-0.999}" --tilt "${TILT:-15}"
      --ckpt-every "${CKPT:-2000}" --resume 1 ${AUG:-} "$@")
lane() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LANE"; }
child=0
trap 'touch "$STOPF"; [ $child -gt 0 ] && kill -TERM $child 2>/dev/null' TERM INT
lane "start host $HOST gpu $GPU bin $U ($(sha1sum "$U" | cut -c1-8)) args ${ARGS[*]}"
n=0
while :; do
  [ -e "$STOPF" ] && { lane "stop file: exit"; exit 0; }
  while [ "$(free -g | awk '/Mem:/ {print $7}')" -lt "${NEEDGB:-12}" ]; do lane "waiting: MemAvailable < ${NEEDGB:-12} GB"; sleep 60; [ -e "$STOPF" ] && exit 0; done
  nice -n "${NICE:-5}" "$U" vwt-train "${ARGS[@]}" >> "$LOG" 2>&1 &
  child=$!; lane "trainer pid $child"
  # the stop file also ends a running trainer (SIGTERM: it saves its last committed step)
  while kill -0 $child 2>/dev/null; do [ -e "$STOPF" ] && { lane "stop file: SIGTERM $child"; kill -TERM $child; wait $child; lane "trainer exited after stop"; exit 0; }; sleep 15; done
  wait $child; rc=$?; child=0
  [ $rc = 0 ] && { lane "trainer finished (rc 0)"; exit 0; }
  n=$((n + 1)); lane "trainer died rc $rc (restart $n / ${MAXRESTART:-20})"
  [ $n -ge "${MAXRESTART:-20}" ] && { lane "too many restarts: exit"; exit 1; }
  sleep 60
done

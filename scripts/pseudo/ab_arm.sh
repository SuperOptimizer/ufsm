#!/usr/bin/env bash
# scripts/pseudo/ab_arm.sh NAME SRCJ SNAP [EXTRA...]: one arm of the pseudo-label A/B (DESIGN 2026-11-17 "Fit-derived pseudo labels") on desk
# card 1, beside try9 on card 0. try9's argv (the last "trainer pid ... argv" line of $TRY/../scratch2.resume.log) with:
#   - --resume from a copy of SNAP (a try9 snapshot: weights + optimizer + step), --out $R/NAME, --steps step(SNAP) + ARMSTEPS (2000:
#     the wsd cooldown 0.2 then covers the arm's last 400 steps, the same in every arm), sources SRCJ;
#   - --det 1 (both arms draw the same labelled windows from the same seed) and a reduced sampler: --workers 3 --sampler-slots 1
#     --onfly-slots 1, --echo-max 0 (echo reuse depends on sampler timing: with it the arms see different windows even under --det 1; lead
#     11-17), so the trainer waits for fresh windows; WORKERS / SLOTS / ONFLYSLOTS / ECHO override (--replay K excludes --echo-max);
#   - --val-every 100000000 (no in-run validation: HOLD / the holdout boxes are scored afterwards by wbench) and --snap-every 500;
#   - no --init-from / --init-widen / --init-lr-* (the snapshot already holds the widened weights and is past the 2000-step ramp).
# Memory: ufsm-train.slice, own scope capped at MEMCAP (40G). Tripwire, every 60 s: the trainer's RssAnon + RssShmem and MemAvailable to
#   $R/NAME.mem.log; MemAvailable < MINAVAIL (20) GB -> SIGTERM the ARM (graceful save), SIGKILL after 120 s: the arm dies, never try9.
# GPU: GPU (1; 0 when try9 is held) exclusively: flock -x on try9's lock set (opt/gpuN.lock, the suite's gpuN.excl / gpuN.0 / gpuN.1), held for the arm's life.
# Env as try9's trainer (MALLOC_ARENA_MAX 4, UFSM_HEAD_SLAB_Z 16, UFSM_HOSTMEM 1, UFSM_SAMPLER_THP 0); LOGE (50) the step-line interval.
# DRY=1 prints the launch line only.
set -uo pipefail
NAME=${1:?NAME}; SRCJ=${2:?SRCJ}; SNAP=${3:?SNAP}; shift 3
R=${R:-/vesuvius/ufsm/v2/lat/runs}; GPU=${GPU:-1}; ARMSTEPS=${ARMSTEPS:-2000}
MEMCAP=${MEMCAP:-40G}; MINAVAIL=${MINAVAIL:-20}; RL=${RL:-$R/scratch2.resume.log}
o=$R/$NAME; [ -e "$o/last.ckpt" ] && [ "${DRY:-0}" != 1 ] && { echo "ab_arm: $o/last.ckpt exists (an earlier start?): remove it to restart from SNAP" >&2; exit 2; }
argv=$(grep 'trainer pid .* argv:' "$RL" | tail -1 | sed 's/.* argv: //'); [ -n "$argv" ] || { echo "ab_arm: no trainer argv in $RL" >&2; exit 2; }
read -r -a A <<< "$argv"
U=${U:-${A[0]}}   # (try9's binary, from its argv)
st=$(head -c 200000 "$SNAP" | head -1 | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2); [ -n "$st" ] || { echo "ab_arm: no step in $SNAP" >&2; exit 2; }
# rebuild the argv: drop the binary, "train", the sources, and the flags replaced below (with their values)
drop=" --log-every --echo-max --out --steps --workers --sampler-slots --onfly-slots --val-every --snap-every --snap-keep-every --snap-keep-last --resume --init-from --init-widen --init-lr-mult --init-lr-steps --det "
N=(); i=3
while [ $i -lt ${#A[@]} ]; do
  a=${A[$i]}
  if [[ "$drop" == *" $a "* ]]; then i=$((i + 2)); continue; fi
  N+=("$a"); i=$((i + 1))
done
ARGS=("$SRCJ" --out "$o" --steps $((st + ARMSTEPS)) "${N[@]}" --log-every ${LOGE:-50} --workers ${WORKERS:-3} --sampler-slots ${SLOTS:-1} --onfly-slots ${ONFLYSLOTS:-1} --echo-max ${ECHO:-0} --det 1
      --val-every 100000000 --snap-every 500 --snap-keep-every 500 --snap-keep-last 8 --resume "$o/last.ckpt" "$@")
if [ "${DRY:-0}" = 1 ]; then echo "CUDA_VISIBLE_DEVICES=$GPU memcap -t -m $MEMCAP $U train ${ARGS[*]}   # from $SNAP (step $st)"; exit 0; fi
mkdir -p "$o"; cp --reflink=auto "$SNAP" "$o/last.ckpt" || exit 1; cp "$SRCJ" "$o/sources.json.first"
fd=7; for lk in /vesuvius/ufsm/opt/gpu$GPU.lock /tmp/ufsm-suite-locks/gpu$GPU.excl /tmp/ufsm-suite-locks/gpu$GPU.0 /tmp/ufsm-suite-locks/gpu$GPU.1; do
  eval "exec $fd> $lk"; flock -x $fd; fd=$((fd + 1)); done   # (try9's own lock set: the run lock, the suite's exclusive lock and both slots)
echo "$(date '+%F %T') $NAME start from $SNAP (step $st): $U train ${ARGS[*]}" | tee -a "$o.launch.log"
CUDA_VISIBLE_DEVICES=$GPU MALLOC_ARENA_MAX=4 UFSM_HEAD_SLAB_Z=${UFSM_HEAD_SLAB_Z:-16} UFSM_HOSTMEM=${UFSM_HOSTMEM:-1} UFSM_SAMPLER_THP=${UFSM_SAMPLER_THP:-0} nice -n 5 /home/forrest/bin/memcap -t -m "$MEMCAP" "$U" train "${ARGS[@]}" >> "$o.log" 2>&1 &
tp=$!; echo $tp > "$o.pid"
( while kill -0 $tp 2>/dev/null; do
    av=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    rss=$(for p in $tp $(pgrep -P $tp) $(pgrep -f "train $SRCJ --out $o " ); do awk '/RssAnon|RssShmem/ {s += $2} END {print s + 0}' /proc/$p/status 2>/dev/null; done | sort -n | tail -1)
    echo "$(date '+%F %T') avail ${av} GB arm rss $((${rss:-0} / 1048576)) GB" >> "$o.mem.log"
    if [ "$av" -lt "$MINAVAIL" ]; then
      echo "$(date '+%F %T') TRIPWIRE: MemAvailable $av GB < $MINAVAIL: SIGTERM the arm" | tee -a "$o.mem.log" "$o.launch.log"
      pkill -TERM -f "train $SRCJ --out $o " ; sleep 120; pkill -KILL -f "train $SRCJ --out $o "; break
    fi
    sleep 60; done ) &
wait $tp; rc=$?
echo "$(date '+%F %T') $NAME exit rc=$rc, last step $(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' "$o.log")" | tee -a "$o.launch.log"

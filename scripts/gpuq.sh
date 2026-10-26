#!/usr/bin/env bash
# gpuq.sh: a small priority queue in front of the laptop GPU lock (~/ufsm-data/gpu.lock). Pure bash + flock.
#
#   gpuq.sh run PRIO TAG cmd args...   wait for the GPU (priority PRIO), run cmd holding it, release (exit code = cmd's)
#   gpuq.sh status                     the holder (from /proc/locks) and the waiting tickets, best first
#   . gpuq.sh                          (sourced) gpuq_acquire PRIO TAG | gpuq_yield | gpuq_release, for scripts that lock per box
#
# PRIO: 0 | now (short checks: pcheck, a quick test), 1 | eval (suites: wgate, fulltest's GPU block: one block, they never yield),
# 2 | bg (long background loops: core benches, slab ceilings, p4unwrap lanes: call gpuq_yield at every box boundary).
# The grant is still an exclusive flock on the lock file, so scripts that call plain `flock ~/ufsm-data/gpu.lock cmd` keep working
# and stay mutually exclusive with queue jobs; they just have no priority (a blocked plain flock may win the lock at a release).
# Order: lowest PRIO first, FIFO (ticket number) inside a priority. Only the head ticket waits on the lock: a blocking flock (the kernel
# serves the longest-blocked waiter first, so a head that re-blocked every few s would lose every release to plain-flock waiters;
# a better newcomer kills the head's flock child, the head re-checks and steps back). Others poll every GPUQ_POLL s.
# gpuq_yield (at a box boundary): eval / now holders release only for a waiting ticket of higher priority and re-queue with their
# old ticket number (first again among their priority); bg holders always let go for 1 s (plain-flock waiters are invisible) and
# re-queue behind waiting bg tickets (round robin). Tickets of dead processes are dropped.
# A command started by `gpuq.sh run` (or by a holder) inherits the lock: nested gpuq_acquire calls return at once
# (GPUQ_HELD=1 in the environment) and nested yields / releases do nothing; wbench.sh's NOLOCK=1 stays valid.
# Env: GPUQ_LOCK (default $LOCK, else ~/ufsm-data/gpu.lock), GPUQ_DIR (tickets; default <lock dir>/gpuq), GPUQ_POLL.

GPUQ_LOCK=${GPUQ_LOCK:-${LOCK:-$HOME/ufsm-data/gpu.lock}}
GPUQ_DIR=${GPUQ_DIR:-$(dirname "$GPUQ_LOCK")/gpuq}
GPUQ_POLL=${GPUQ_POLL:-0.5}

gpuq_prio() { case $1 in 0|now) echo 0;; 1|eval) echo 1;; 2|bg) echo 2;; *) echo "gpuq: priority $1 (0|now 1|eval 2|bg)" >&2; return 1;; esac; }
gpuq_log() { echo "$(date '+%m-%d %H:%M:%S') gpuq: $*" >&2; }
# live tickets "prio seq pid tag", best first (dead owners' tickets removed)
gpuq_tickets() {
  local f p s q
  for f in "$GPUQ_DIR"/t.*; do [ -e "$f" ] || continue
    IFS=. read -r _ p s q <<< "${f##*/}"
    if ! kill -0 "$q" 2> /dev/null; then rm -f "$f"; continue; fi
    echo "$p $s $q $(head -c 200 "$f" 2> /dev/null)"
  done | sort -k1,1n -k2,2n
}
# (/proc/locks cannot tell who holds or waits for the lock: it hides locks of processes in other pid namespaces, e.g. sandboxes)
gpuq_busy() { local fd r=1; exec {fd}> "$GPUQ_LOCK" || return 1; flock -n $fd && r=1 || r=0; exec {fd}>&-; return $r; }

gpuq_acquire() {   # gpuq_acquire PRIO TAG
  [ "${GPUQ_HELD:-0}" = 1 ] && return 0
  local pr; pr=$(gpuq_prio "${1:-2}") || return 2; local tag=${2:-job}
  mkdir -p "$GPUQ_DIR"; : >> "$GPUQ_LOCK"
  local seq=${GPUQ_SEQ:-}
  if [ -z "$seq" ]; then   # a new ticket number (counter under its own lock)
    seq=$( { flock 8; n=$(( $(cat "$GPUQ_DIR/seq" 2> /dev/null || echo 0) + 1 )); echo $n > "$GPUQ_DIR/seq"; printf '%09d' $n; } 8> "$GPUQ_DIR/seq.lock" )
  fi
  local me=$BASHPID t="$GPUQ_DIR/t.$pr.$seq.$BASHPID" said=0 head
  echo "$tag" > "$t"
  local hp hpr wp ok
  # a better ticket than the head that is blocked on the lock interrupts its wait (the head then re-checks and steps back)
  if [ -s "$GPUQ_DIR/headwait" ] && read -r hp hpr < "$GPUQ_DIR/headwait" && [ "$pr" -lt "${hpr:-0}" ]; then kill "$hp" 2> /dev/null; fi
  while :; do
    head=$(gpuq_tickets | head -1)
    if [ "${head%% *}" = "$pr" ] && [ "$(echo "$head" | cut -d' ' -f2)" = "$seq" ]; then
      exec {GPUQ_FD}> "$GPUQ_LOCK"
      # the head blocks like a plain flock (a polling head would lose every release to blocked plain flocks), in a child whose pid
      # is published so that a better newcomer can cut the wait short (the lock, once taken, belongs to the shared open file)
      flock -w "${GPUQ_WAIT:-3600}" $GPUQ_FD & wp=$!; echo "$wp $pr" > "$GPUQ_DIR/headwait"; wait $wp && ok=1 || ok=0
      [ "$(cat "$GPUQ_DIR/headwait" 2> /dev/null)" = "$wp $pr" ] && rm -f "$GPUQ_DIR/headwait"
      if [ $ok = 1 ]; then
        rm -f "$t"; GPUQ_HELD=1; GPUQ_OWNER=$me; GPUQ_PR=$pr; GPUQ_TAG=$tag; GPUQ_SEQ=$seq; export GPUQ_HELD
        echo "$me $pr $tag $(date +%s)" > "$GPUQ_DIR/holder"
        [ $said = 1 ] && gpuq_log "$tag got the GPU"
        return 0
      fi
      exec {GPUQ_FD}>&-
      [ $said = 0 ] && { gpuq_log "$tag (prio $pr) waiting for the GPU"; said=1; }
      continue   # (re-check the head: a better ticket may have come)
    fi
    [ $said = 0 ] && { gpuq_log "$tag (prio $pr) waiting for the GPU"; said=1; }
    sleep "$GPUQ_POLL"
  done
}
gpuq_release() {
  [ "${GPUQ_HELD:-0}" = 1 ] && [ "${GPUQ_OWNER:-}" = "$BASHPID" ] || return 0
  rm -f "$GPUQ_DIR/holder"; flock -u $GPUQ_FD; exec {GPUQ_FD}>&-; GPUQ_HELD=0; export GPUQ_HELD
}
gpuq_yield() {   # at a box boundary: give the GPU to a better waiter, then wait for it again
  [ "${GPUQ_HELD:-0}" = 1 ] && [ "${GPUQ_OWNER:-}" = "$BASHPID" ] || return 0
  local best; best=$(gpuq_tickets | head -1 | cut -d' ' -f1)
  # bg always lets go for a moment (plain-flock waiters are invisible), re-queued behind waiting bg tickets; eval / now only for better tickets
  if [ "$GPUQ_PR" != 2 ] && ! { [ -n "$best" ] && [ "$best" -lt "$GPUQ_PR" ]; }; then return 0; fi
  local pr=$GPUQ_PR tag=$GPUQ_TAG seq=$GPUQ_SEQ
  gpuq_release; [ "$pr" = 2 ] && [ -n "$best" ] && [ "$best" = 2 ] && seq=
  sleep 1   # a blocked plain flock takes the lock now
  GPUQ_SEQ=$seq gpuq_acquire $pr "$tag"
}

gpuq_status() {
  local p pr tag since
  if gpuq_busy; then
    if [ -s "$GPUQ_DIR/holder" ] && read -r p pr tag since < "$GPUQ_DIR/holder" && kill -0 "$p" 2> /dev/null; then
      echo "holder: pid $p prio $pr $tag (since $(date -d @$since '+%m-%d %H:%M:%S'))"
    else echo "holder: a plain flock user (not queued)"; fi
  else echo "holder: none"; fi
  gpuq_tickets | awk '{printf "  prio %s ticket %d pid %s %s\n", $1, $2, $3, $4}'
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case ${1:-} in
    run) [ $# -ge 4 ] || { echo "usage: gpuq.sh run PRIO TAG cmd args..." >&2; exit 2; }
         pr=$2 tag=$3; shift 3; gpuq_acquire "$pr" "$tag" || exit 2; "$@"; rc=$?; gpuq_release; exit $rc;;
    status) gpuq_status;;
    *) sed -n '2,20p' "$0"; exit 2;;
  esac
fi

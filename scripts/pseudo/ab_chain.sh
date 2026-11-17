#!/usr/bin/env bash
# scripts/pseudo/ab_chain.sh SRCA SRCB: tonight's paired pseudo-label A/B on card 0 while try9 is held (lead 11-17, option 1). Run it as
#   S2CTL_OWNER=pseudolab bash scripts/run/scratch2/s2ctl.sh hold "pseudo A/B" -- bash scripts/pseudo/ab_chain.sh SRCA SRCB
# so s2ctl stops try9 before and resumes it after, whatever this returns. Steps (each must pass or the chain ends, and try9 resumes):
#   0. the base: a reflink copy of try9's last.ckpt at the stop ($R/pab-base.ckpt): both arms and the base HOLD eval start from it;
#   1. --det check: arm A twice for DETN (20) windows, --log-every 1: the step lines up to " qk " (losses, accuracies, gauges) must be
#      byte-identical, else the pairing is not real and the chain stops;
#   2. arm A (SRCA) then arm B (SRCB), ARMSTEPS (2000) each, try9's full sampler (12 slots / 10 workers / 3 onfly slots), echo 0, --det 1,
#      the train slice with MEMCAP (136G, try9's own) and the MemAvailable tripwire (scripts/pseudo/ab_arm.sh).
set -uo pipefail
SRCA=${1:?SRCA}; SRCB=${2:?SRCB}; H=$(cd "$(dirname "$0")/../.." && pwd); R=${R:-/vesuvius/ufsm/v2/lat/runs}
export GPU=0 WORKERS=${WORKERS:-10} SLOTS=${SLOTS:-12} ONFLYSLOTS=${ONFLYSLOTS:-3} ECHO=0 MEMCAP=${MEMCAP:-136G}
L=$R/pab-chain.log; log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
B=$R/pab-base.ckpt
[ -s "$B" ] || cp --reflink=auto "$R/scratch2/last.ckpt" "$B" || { log "cannot copy the base"; exit 1; }
log "base $B (step $(head -c 200000 "$B" | head -1 | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2)); A $SRCA, B $SRCB"
# 1. the --det check
for k in 1 2; do
  d=$R/pab-det$k; case "$d" in "$R"/pab-det[12]) rm -rf -- "$d" "$d".log "$d".launch.log "$d".mem.log "$d".pid;; esac
  ARMSTEPS=${DETN:-20} LOGE=1 bash "$H/scripts/pseudo/ab_arm.sh" pab-det$k "$SRCA" "$B" || { log "det run $k failed"; exit 1; }
done
a=$(grep -E '^step +[0-9]+ ' "$R/pab-det1.log" | sed 's/ qk .*//'); b=$(grep -E '^step +[0-9]+ ' "$R/pab-det2.log" | sed 's/ qk .*//')
n=$(echo "$a" | grep -c .)
if [ "$n" -lt 5 ] || [ "$a" != "$b" ]; then log "DET CHECK FAILED: $n step lines, $(diff <(echo "$a") <(echo "$b") | grep -c '^<') differ: no paired A/B (see pab-det1.log / pab-det2.log)"; exit 3; fi
log "det check: $n step lines byte-identical up to qk"
# 2. the arms
for arm in A B; do
  src=$SRCA; [ $arm = B ] && src=$SRCB
  bash "$H/scripts/pseudo/ab_arm.sh" pab-$arm "$src" "$B" || { log "arm $arm failed"; exit 1; }
  log "arm $arm: $(grep -E '^step +[0-9]+ ' "$R/pab-$arm.log" | tail -1 | cut -c1-120)"
done
log "chain done"

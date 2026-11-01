#!/usr/bin/env bash
# scripts/v2/snap.sh [NLANES]: snap the v2 training wraps (scripts/unroll/snap_existing.sh) over V2/scrolls.tsv rows, reading local CT only.
# A row runs when: split = train, ct_root set (the fetched scan is complete), no V2/wraps_snap/<S>/done, and either fields are set
# (target mix) or <S> is listed in V2/snap.ct (target ct: scrolls whose fields never come back). Rows with no fields yet wait.
# Lanes claim scrolls with flock; a lane only starts a scroll while MemAvailable > MINMEM GB (40). Output V2/wraps_snap/<S>/wraps_snap
# + snap.log; MEASURE=0 (no surf-stats). Re-run scripts/v2/scrolls.sh to pick up new CT / fields, then this again (done scrolls skip).
# Env: V2 (/vesuvius/ufsm/v2), U (V2/src/build/ufsm), TH (threads per snap, 10), MINMEM, ONLY ("S1 S2": restrict).
set -u
NL=${1:-3}; H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $H/../.. && pwd)
V2=${V2:-/vesuvius/ufsm/v2}; U=${U:-$V2/src/build/ufsm}; O=$V2/wraps_snap; mkdir -p $O $V2/logs
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $V2/logs/snap.log; }
memok() { [ $(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo) -gt ${MINMEM:-40} ]; }
lane() {
  local S SPLIT NW WR WC SJ SL AX CR K UM LV FL PK T
  tail -n +2 $V2/scrolls.tsv | while IFS=$'\t' read -r S SPLIT NW WR WC SJ SL AX CR K UM LV FL PK _; do
    [ "$SPLIT" = train ] && [ "$CR" != - ] && [ "$AX" != - ] || continue
    [ -n "${ONLY:-}" ] && { case " $ONLY " in *" $S "*) ;; *) continue;; esac; }
    [ -e $O/$S/done ] && continue
    T=; [ "$FL" != - ] && T=mix; [ -z "$T" ] && grep -qx $S $V2/snap.ct 2>/dev/null && T=ct; [ -n "$T" ] || continue
    [ -e $CR/$K/$LV/zarr.json ] || { log "$S: no local CT $CR/$K/$LV"; continue; }
    mkdir -p $O/$S; exec 9> $O/$S/claim; flock -n 9 || continue; [ -e $O/$S/done ] && continue
    until memok; do sleep 60; done
    log "$S: start (target $T, $NW wraps)"
    if RUN=$V2/wraps_src/$S CTROOT=$CR SNAPT=$T MEASURE=0 TH=${TH:-10} U=$U nice -n 10 bash $REPO/scripts/unroll/snap_existing.sh $S $O > $O/$S/driver.log 2>&1 &&
       [ $(ls $O/$S/wraps_snap | grep -c '\.sfc$') = $NW ]; then
      echo "$T $(date '+%F %T')" > $O/$S/done; log "$S: done ($(tail -n 1 $O/$S/driver.log))"
    else log "$S: FAILED (see $O/$S/snap.log)"; fi
  done
}
for l in $(seq 1 $NL); do lane & sleep 5; done; wait
log "snap pass finished"

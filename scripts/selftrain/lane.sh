#!/usr/bin/env bash
# lane.sh KIND: one restart-safe worker lane of a self-training round (A/r$R). KIND pred = teacher fields for every box (GPU); KIND fit =
# fit + label every box whose fields exist (GPU, or CPU=1). Boxes claimed by mkdir (A/r$R/claim/<KIND>.<box>); done markers
# A/r$R/done/<KIND>.<box>; failures A/r$R/done/<KIND>.<box>.fail (not retried until deleted). Order: scans.tsv rows with state "go",
# box 0 of every scan first, then box 1, ... (all scans get labels early). Stops on A/r$R/STOP; KIND fit waits for fields while the pred
# lane runs (A/r$R/pred.alldone when it has finished). Keeps >= MINFREE_GB (40) of MemAvailable before each fit.
# Env: as box.sh (A, R, CK, U, GPU, CPU, TH, ARMS), ONLY (space-separated scroll list), MINFREE_GB, SKIP (box.sh STAGEs to skip)
set -u
K=$1; A=${A:-/vesuvius/ufsm/selftrain}; R=${R:-0}; RD=$A/r$R; D=$(dirname "$(readlink -f "$0")"); export A R
mkdir -p $RD/claim $RD/done
list() {   # "SCROLL I" in round-robin order
  local mx=$(awk -F'\t' '!/^#/ && $8 == "go" {print $7}' $D/scans.tsv | sort -n | tail -1)
  for i in $(seq 0 $((mx - 1))); do
    awk -F'\t' -v i=$i -v only="${ONLY:-}" '!/^#/ && $8 == "go" && i < $7 && (only == "" || index(" " only " ", " " $1 " ")) {print $1, i}' $D/scans.tsv
  done
}
while :; do
  any=0; waiting=0
  while read -r S I; do
    [ -e $RD/STOP ] && { echo "$(date +%H:%M) STOP"; exit 0; }
    [ -s $A/boxes/$S.json ] && [ "$(jq length $A/boxes/$S.json)" -gt $I ] || continue
    N=$S-b$I; [ -e $RD/done/$K.$N ] || [ -e $RD/done/$K.$N.fail ] && continue
    if [ $K = fit ] && [ ! -e $RD/done/pred.$N ]; then waiting=1; continue; fi
    mkdir $RD/claim/$K.$N 2> /dev/null || continue
    if [ $K = fit ]; then
      while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt ${MINFREE_GB:-40} ]; do sleep 30; done
      bash $D/box.sh fit $S $I && bash $D/box.sh label $S $I; rc=$?
    else bash $D/box.sh pred $S $I; rc=$?; fi
    if [ $rc = 0 ]; then touch $RD/done/$K.$N; else touch $RD/done/$K.$N.fail; fi
    rmdir $RD/claim/$K.$N; any=1
  done < <(list)
  [ $K = pred ] && [ $any = 0 ] && { touch $RD/pred.alldone; echo "$(date +%H:%M) pred: all done"; exit 0; }
  if [ $K = fit ] && [ $any = 0 ]; then
    [ $waiting = 0 ] && [ -e $RD/pred.alldone ] && { echo "$(date +%H:%M) fit: all done"; exit 0; }
    sleep 60
  fi
done

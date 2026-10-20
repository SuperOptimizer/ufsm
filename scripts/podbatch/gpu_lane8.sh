#!/usr/bin/env bash
# gpu_lane8.sh ID: one GPU-lane worker of fit-field batch v8 (2026-10-16 evening; v7 for a 16 GB RTX 5080 pod with a cgroup v2
# 31 GB / 23.8 CPU limit). Runs predict --fit-zarr from LOCAL CT (prefetched by pf_lane.sh into /workspace/vol) for the lines of
# queue.list, in order. Budgets (env): BUD host RAM (26 GB), GBUD GPU (14.5 GB), WINS the windows tried in order (640; 512 is never
# worth it: its stride of 3 shards makes the field slab 384 cells high without the ring), RETRY the retry window (384), TH predict
# threads (18).
# Memory: a running job reserves its host RAM estimate in resv/<scroll> (10 planes x FZ x min(Y/2, 3 FZ) x X/2 + 6 GB: predict holds a
# ring of 3 rows of FZ cells, FZ = 384 cells at window 896, 256 at 640, 128 at 384) and its GPU estimate in gresv/<scroll> (window 640:
# 13 GB measured on a 96 GB card; GEST640 overrides). A job starts when the reservations, and RSS + shmem (cgroup v2 anon + shmem, v1
# total_rss + total_shmem), fit BUD and GBUD. A line behind the head (the first unfinished, unclaimed line) starts only if it also fits
# beside the head's own target, so the head never waits on it. One retry at RETRY. Axis: axis/new/<s>.json (ufsm axis centroids), else
# axis/<s>.umbilicus.json, else axis/<s>.json (only files in the scan's own frame are kept in axis/). State (restart-safe):
# state/<s>.claim "<worker pid> <predict pid> <window>", .predicted, .uploaded, .failed. After predict: the local CT is deleted and
# up_one.sh (repack + upload) runs detached, one upload at a time (upload.lock).
set -u; ID=$1; cd /workspace; U=/workspace/ufsm/build/ufsm; CK=/workspace/ckpt/prod96.ckpt; ST=out/STATUS.md
BUD=${BUD:-26000000000}; GBUD=${GBUD:-14500000000}; WINS=${WINS:-640}; RETRY=${RETRY:-384}; TH=${TH:-18}
mkdir -p out scratch resv gresv state
log() { ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) g$ID $*" >> $ST ) 8> status.lock; }
used() { if [ -f /sys/fs/cgroup/memory.stat ]; then awk '$1=="anon" || $1=="shmem" {s += $2} END {printf "%.0f", s}' /sys/fs/cgroup/memory.stat
         else awk '$1=="total_rss" || $1=="total_shmem" {s += $2} END {printf "%.0f", s}' /sys/fs/cgroup/memory/memory.stat; fi; }
sum() { cat $1/* 2>/dev/null | awk '{s += $1} END {printf "%.0f", s}'; }
est() { awk -v fz=$1 -v y=$2 -v x=$3 'BEGIN {fy = int((y + 1) / 2); if (fy > 3 * fz) fy = 3 * fz; printf "%.0f", 10 * fz * fy * int((x + 1) / 2) + 6e9}'; }
gest() { case $1 in 896) echo 33000000000;; 640) echo ${GEST640:-13000000000};; *) echo 4000000000;; esac; }
fz_of() { case $1 in 896) echo 384;; 640) echo 256;; 384) echo 128;; esac; }
vdir() { echo vol/$1/volumes/$2.zarr/$3; }
shape() { jq -r '"\(.shape[1]) \(.shape[2])"' shapes/$1.json; }
axis_of() { local f; for f in axis/new/$1.json axis/$1.umbilicus.json axis/$1.json; do [ -f $f ] && { echo $f; return 0; }; done; return 1; }
fin() { [ -e state/$1.predicted ] || [ -e state/$1.uploaded ] || [ -e state/$1.failed ]; }
live() { local a b; read -r a b _ < state/$1.claim 2>/dev/null || return 1; { [ "${a:-0}" != 0 ] && kill -0 $a 2>/dev/null; } || { [ "${b:-0}" != 0 ] && kill -0 $b 2>/dev/null; }; }
pick() { ( flock 9
    now=$(date +%s); last=$(cat last_start 2>/dev/null || echo 0); [ $(( now - last )) -lt 30 ] && exit
    fr=$(( BUD - $(sum resv) )); u=$(( BUD - $(used) )); [ $u -lt $fr ] && fr=$u; fg=$(( GBUD - $(sum gresv) )); he=""; n=0
    while read -r s scan um lv <&3; do
      fin $s && continue
      if [ -e state/$s.claim ]; then live $s && continue; log "$s: stale claim removed ($(cat state/$s.claim))"; rm -f state/$s.claim resv/$s gresv/$s; fi
      n=$(( n + 1 )); [ $n -gt 16 ] && break
      read -r Y X <<< "$(shape $s)"
      if [ -z "$he" ]; then hd=1; he=0; for w in $WINS; do e=$(est $(fz_of $w) $Y $X); [ $e -le $BUD ] && { he=$e; break; }; done; else hd=0; fi
      [ -e $(vdir $s $scan $lv)/.complete ] || continue
      axis_of $s > /dev/null || continue
      for w in $WINS; do e=$(est $(fz_of $w) $Y $X); g=$(gest $w)
        [ $e -le $fr ] && [ $g -le $fg ] || continue
        [ $hd = 1 ] || [ $(( e + he )) -le $BUD ] || continue
        echo "$$ 0 $w" > state/$s.claim; echo $e > resv/$s; echo $g > gresv/$s; echo $now > last_start
        echo "$s $scan $um $lv|$w|$e|$Y|$X"; exit
      done
    done 3< queue.list ) 9> queue.lock; }
run() {   # s scan um lv w -> rc; peak RSS in out/$s/time-$w.txt
  local s=$1 scan=$2 um=$3 lv=$4 w=$5 ax o p rc; ax=$(axis_of $s) || return 98
  o=out/$s/${scan%%-*}-fitfields-20261016-ufsm-prod96-h2.zarr; mkdir -p out/$s; rm -rf $o scratch/$s-p
  UFSM_PRED_PROF=1 /usr/bin/time -f %M -o out/$s/time-$w.txt $U predict $CK /workspace/vol $s/volumes/$scan.zarr scratch/$s-p --um $um --level $lv \
    --window $w --halo 64 --shard 128 --ema 1 --q 8 --threads $TH --gpu 0 --axis $ax --levels 1 --fit-zarr $o --fit-h 2 --fit-q 8 --no-out 1 \
    > out/$s/predict-$w.log 2>&1 &
  p=$!; echo "$$ $p $w" > state/$s.claim; wait $p; rc=$?
  [ $rc = 0 ] && [ ! -f $o/fields.json ] && rc=99; return $rc; }
release() { rm -f state/$1.claim resv/$1 gresv/$1; }
log "worker start (pid $$, v8: BUD $(( BUD / 1000000000 )) GB, GBUD $(( GBUD / 1000000000 )) GB, windows $WINS retry $RETRY, $TH threads)"
while :; do
  r=$(pick)
  if [ -z "$r" ]; then
    rem=$(while read -r s _; do fin $s || [ -e state/$s.claim ] || echo $s; done < queue.list | wc -l)
    [ $rem = 0 ] && { log "nothing left to start, worker done"; break; }
    sleep 20; continue
  fi
  IFS="|" read -r line w e Y X <<< "$r"; set -- $line; s=$1 scan=$2 um=$3 lv=$4
  log "$s start: window $w, axis $(axis_of $s), est RAM $(( e / 1000000000 )) GB GPU $(( $(gest $w) / 1000000000 )) GB (cross-section ${Y}x${X} level $lv); reserved RAM $(( $(sum resv) / 1000000000 )) GB GPU $(( $(sum gresv) / 1000000000 )) GB"
  t0=$(date +%s); run $s $scan $um $lv $w; rc=$?
  if [ $rc != 0 ]; then
    w2=0; [ $w != $RETRY ] && w2=$RETRY
    log "$s FAILED rc $rc at window $w after $(( $(date +%s) - t0 ))s (peak $(tail -1 out/$s/time-$w.txt 2>/dev/null) KB): $(tail -c 300 out/$s/predict-$w.log | tr '\n' ' ' | cut -c1-200)"
    [ $w2 = 0 ] && { touch state/$s.failed; release $s; continue; }
    e2=$(est $(fz_of $w2) $Y $X); g2=$(gest $w2)
    until ( flock 9; oth=$(( $(sum resv) - $(cat resv/$s) )); goth=$(( $(sum gresv) - $(cat gresv/$s) ))
            [ $(( oth + e2 )) -le $BUD ] && [ $(( goth + g2 )) -le $GBUD ] && [ $(( $(used) + e2 )) -le $BUD ] && echo $e2 > resv/$s && echo $g2 > gresv/$s ) 9> queue.lock
    do sleep 20; done
    log "$s retrying at window $w2"; t0=$(date +%s); run $s $scan $um $lv $w2; rc=$?; w=$w2
    [ $rc != 0 ] && { log "$s FAILED again rc $rc at window $w; giving up"; touch state/$s.failed; release $s; continue; }
  fi
  tp=$(( $(date +%s) - t0 )); pk=$(( $(tail -1 out/$s/time-$w.txt) / 1000000 ))
  log "$s predicted in ${tp}s at window $w ($(awk -v v=$(jq -r '.shape | .[0] * .[1] * .[2]' shapes/$s.json) -v t=$tp 'BEGIN {printf "%.0f", v / t / 1e6}') Mvox/s of the volume), peak RSS ${pk} GB"
  touch state/$s.predicted; release $s; rm -rf vol/$s scratch/$s-p
  TAG=up setsid nohup flock -o upload.lock ./up_one.sh $s $scan $um $lv $w > out/$s/up.log 2>&1 < /dev/null &
done

#!/usr/bin/env bash
# gpu_lane7.sh ID: one GPU-lane worker of fit-field batch v7 (2026-10-16; v5 + predict's 3-row field slab ring, 7dafc9e); run 3 of them. Each runs predict --fit-zarr from LOCAL CT
# (prefetched by pf_lane.sh into /workspace/vol; reading over HTTPS range requests left the GPU idle) for the lines of queue.list, in order.
# Memory: a running job reserves its host RAM estimate in resv/<scroll> (10 planes x FZ x min(Y/2, 3 FZ) x X/2 + 6 GB: predict holds a ring
# of 3 rows of FZ cells, FZ = 384 cells at window 896, 256 at 640, 128 at 384; before the ring the whole Y/2: 0009B 62.7 GB measured) and its GPU estimate in gresv/<scroll> (31 GB
# measured at window 896, ~window^3). A job starts when the reservations, and RSS + shmem, fit 112 GB of the 125 GB cgroup and 90 GB of
# the 96 GB GPU (two jobs at 896 + one at 640: 33 + 33 + 11 GB measured), at window 896, else 640 (384 only as the retry); the third job
# covers the others' air stretches (read, no GPU work). A line behind the head (the first unfinished,
# unclaimed line) starts only if it also fits beside the head's own target, so the head never waits on it. One retry at the next smaller
# window with 24 threads. State (restart-safe): state/<s>.claim "<worker pid> <predict pid> <window>", .predicted, .uploaded, .failed.
# After predict: the local CT is deleted and up_one.sh (repack + upload) runs detached, one upload at a time (upload.lock).
set -u; ID=$1; cd /workspace; U=/workspace/ufsm/build/ufsm; CK=/workspace/ckpt/prod96.ckpt; ST=out/STATUS.md
BUD=112000000000; GBUD=90000000000; mkdir -p out scratch resv gresv state
log() { ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) g$ID $*" >> $ST ) 8> status.lock; }
used() { awk '$1=="total_rss" || $1=="total_shmem" {s += $2} END {printf "%.0f", s}' /sys/fs/cgroup/memory/memory.stat; }
sum() { cat $1/* 2>/dev/null | awk '{s += $1} END {printf "%.0f", s}'; }
est() { awk -v fz=$1 -v y=$2 -v x=$3 'BEGIN {fy = int((y + 1) / 2); if (fy > 3 * fz) fy = 3 * fz; printf "%.0f", 10 * fz * fy * int((x + 1) / 2) + 6e9}'; }
gest() { case $1 in 896) echo 33000000000;; 640) echo 13000000000;; *) echo 4000000000;; esac; }
fz_of() { case $1 in 896) echo 384;; 640) echo 256;; 384) echo 128;; esac; }
vdir() { echo vol/$1/volumes/$2.zarr/$3; }
shape() { jq -r '"\(.shape[1]) \(.shape[2])"' shapes/$1.json; }
fin() { [ -e state/$1.predicted ] || [ -e state/$1.uploaded ] || [ -e state/$1.failed ]; }
live() { local a b; read -r a b _ < state/$1.claim 2>/dev/null || return 1; { [ "${a:-0}" != 0 ] && kill -0 $a 2>/dev/null; } || { [ "${b:-0}" != 0 ] && kill -0 $b 2>/dev/null; }; }
pick() { ( flock 9
    now=$(date +%s); last=$(cat last_start 2>/dev/null || echo 0); [ $(( now - last )) -lt 30 ] && exit
    fr=$(( BUD - $(sum resv) )); u=$(( BUD - $(used) )); [ $u -lt $fr ] && fr=$u; fg=$(( GBUD - $(sum gresv) )); he=""; n=0; nrun=$(ls gresv | wc -l)
    while read -r s scan um lv <&3; do
      fin $s && continue
      if [ -e state/$s.claim ]; then live $s && continue; log "$s: stale claim removed ($(cat state/$s.claim))"; rm -f state/$s.claim resv/$s gresv/$s; fi
      n=$(( n + 1 )); [ $n -gt 16 ] && break
      read -r Y X <<< "$(shape $s)"
      if [ -z "$he" ]; then hd=1; he=0; for w in 896 640; do e=$(est $(fz_of $w) $Y $X); [ $e -le $BUD ] && { he=$e; break; }; done; else hd=0; fi
      [ -e $(vdir $s $scan $lv)/.complete ] || continue
      for w in 896 640; do e=$(est $(fz_of $w) $Y $X); g=$(gest $w)
        [ $e -le $fr ] && [ $g -le $fg ] || continue
        [ $hd = 1 ] || [ $(( e + he )) -le $BUD ] || continue
        echo "$$ 0 $w" > state/$s.claim; echo $e > resv/$s; echo $g > gresv/$s; echo $now > last_start
        echo "$s $scan $um $lv|$w|$e|$Y|$X"; exit
      done
    done 3< queue.list ) 9> queue.lock; }
run() {   # s scan um lv w -> rc; peak RSS in out/$s/time-$w.txt
  local s=$1 scan=$2 um=$3 lv=$4 w=$5 ax o p rc; ax=axis/new/$s.json; [ -f axis/$s.umbilicus.json ] && ax=axis/$s.umbilicus.json
  [ -f axis/$s.json ] && [ ! -f axis/new/$s.json ] && ax=axis/$s.json
  o=out/$s/${scan%%-*}-fitfields-20261016-ufsm-prod96-h2.zarr; mkdir -p out/$s; rm -rf $o scratch/$s-p
  UFSM_PRED_PROF=1 /usr/bin/time -f %M -o out/$s/time-$w.txt $U predict $CK /workspace/vol $s/volumes/$scan.zarr scratch/$s-p --um $um --level $lv \
    --window $w --halo 64 --shard 128 --ema 1 --q 8 --threads ${TH:-32} --gpu 0 --axis $ax --levels 1 --fit-zarr $o --fit-h 2 --fit-q 8 --no-out 1 \
    > out/$s/predict-$w.log 2>&1 &
  p=$!; echo "$$ $p $w" > state/$s.claim; wait $p; rc=$?
  [ $rc = 0 ] && [ ! -f $o/fields.json ] && rc=99; return $rc; }
release() { rm -f state/$1.claim resv/$1 gresv/$1; }
log "worker start (pid $$, v7)"
while :; do
  r=$(pick)
  if [ -z "$r" ]; then
    rem=$(while read -r s _; do fin $s || [ -e state/$s.claim ] || echo $s; done < queue.list | wc -l)
    [ $rem = 0 ] && { log "nothing left to start, worker done"; break; }
    sleep 20; continue
  fi
  IFS="|" read -r line w e Y X <<< "$r"; set -- $line; s=$1 scan=$2 um=$3 lv=$4
  log "$s start: window $w, est RAM $(( e / 1000000000 )) GB GPU $(( $(gest $w) / 1000000000 )) GB (cross-section ${Y}x${X} level $lv); reserved RAM $(( $(sum resv) / 1000000000 )) GB GPU $(( $(sum gresv) / 1000000000 )) GB"
  t0=$(date +%s); run $s $scan $um $lv $w; rc=$?
  if [ $rc != 0 ]; then
    w2=0; [ $w = 896 ] && w2=640; [ $w = 640 ] && w2=384
    log "$s FAILED rc $rc at window $w after $(( $(date +%s) - t0 ))s (peak $(tail -1 out/$s/time-$w.txt 2>/dev/null) KB): $(tail -c 300 out/$s/predict-$w.log | tr '\n' ' ' | cut -c1-200)"
    [ $w2 = 0 ] && { touch state/$s.failed; release $s; continue; }
    e2=$(est $(fz_of $w2) $Y $X); g2=$(gest $w2)
    until ( flock 9; oth=$(( $(sum resv) - $(cat resv/$s) )); goth=$(( $(sum gresv) - $(cat gresv/$s) ))
            [ $(( oth + e2 )) -le $BUD ] && [ $(( goth + g2 )) -le $GBUD ] && [ $(( $(used) + e2 )) -le $BUD ] && echo $e2 > resv/$s && echo $g2 > gresv/$s ) 9> queue.lock
    do sleep 20; done
    log "$s retrying at window $w2"; t0=$(date +%s); TH=24 run $s $scan $um $lv $w2; rc=$?; w=$w2
    [ $rc != 0 ] && { log "$s FAILED again rc $rc at window $w; giving up"; touch state/$s.failed; release $s; continue; }
  fi
  tp=$(( $(date +%s) - t0 )); pk=$(( $(tail -1 out/$s/time-$w.txt) / 1000000 ))
  log "$s predicted in ${tp}s at window $w ($(awk -v v=$(jq -r '.shape | .[0] * .[1] * .[2]' shapes/$s.json) -v t=$tp 'BEGIN {printf "%.0f", v / t / 1e6}') Mvox/s of the volume), peak RSS ${pk} GB"
  touch state/$s.predicted; release $s; rm -rf vol/$s scratch/$s-p
  TAG=up setsid nohup flock -o upload.lock ./up_one.sh $s $scan $um $lv $w > out/$s/up.log 2>&1 < /dev/null &
done

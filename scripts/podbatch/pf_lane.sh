#!/usr/bin/env bash
# pf_lane.sh: the prefetch lane of fit-field batch v5 (2026-10-16). Downloads the CT level of the lines of queue.list, in order, to
# /workspace/vol (prefetch.sh: whole shard files, 48 HTTP/1.1 connections), up to NPF at a time, while the disk keeps > 150 GB free and
# at most AHEAD volumes are downloading or downloaded but not yet started. Restart-safe: state/<s>.pf (downloader pid), the volume's
# <level>/.complete, and gpu_lane.sh's state/<s>.{claim,predicted,uploaded,failed}.
set -u; cd /workspace; NPF=${NPF:-2}; AHEAD=${AHEAD:-4}; ST=out/STATUS.md; mkdir -p state
log() { ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) pf $*" >> $ST ) 8> status.lock; }
fin() { [ -e state/$1.predicted ] || [ -e state/$1.uploaded ] || [ -e state/$1.failed ]; }
log "lane start (pid $$)"
while :; do
  running=0; ready=0; next=""
  while read -r s scan um lv; do
    fin $s && continue; d=vol/$s/volumes/$scan.zarr/$lv
    if [ -e $d/.complete ]; then [ -e state/$s.claim ] || ready=$(( ready + 1 )); continue; fi
    [ -s state/$s.pf ] && kill -0 $(cat state/$s.pf) 2>/dev/null && { running=$(( running + 1 )); continue; }
    [ -e state/$s.claim ] && continue
    [ -z "$next" ] && next="$s $scan $lv"
  done < queue.list
  [ -z "$next" ] && [ $running = 0 ] && { log "nothing left to prefetch, lane done"; break; }
  free=$(df -B1 --output=avail /workspace | tail -1)
  if [ -n "$next" ] && [ $running -lt $NPF ] && [ $(( running + ready )) -lt $AHEAD ] && [ $free -gt 150000000000 ]; then
    set -- $next
    setsid nohup bash -c 't0=$(date +%s); ./prefetch.sh $1 $2 $3 48 > out/prefetch-$1.log 2>&1; rc=$?; dt=$(( $(date +%s) - t0 + 1 ))
      b=$(du -sb vol/$1 | cut -f1); ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) pf $1 prefetched (rc $rc): $(( b / 1000000 )) MB in ${dt}s, $(( b / dt / 1000000 )) MB/s" >> out/STATUS.md ) 8> status.lock' \
      pf $1 $2 $3 < /dev/null > /dev/null 2>&1 &
    echo $! > state/$1.pf; log "$1 prefetch started (pid $!, $(( free / 1000000000 )) GB free, $running running, $ready ready)"; sleep 3; continue
  fi
  sleep 20
done

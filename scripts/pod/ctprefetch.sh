#!/bin/bash
# scripts/pod/ctprefetch.sh SOURCES.json [P] (2026-11-25, lead: the standard first step of every pod launch on mirror CT).
# Pod data wait is cold-chunk HTTPS latency, not CPU: per-chunk reads from the dl.ash2txt mirror cost ~0.35 s first-byte each (S2E waited
# 30-71 % at 30-86 Mvox/s; DINO2 with z3-fetched slabs 0 %; the E0-E5 dev runs on a warm cache 0-10 %). So before the trainer starts, fetch
# every CT chunk its sampler draws (ufsm prefetch: each source's draw cells at the levels it trains on, the window P around them, the probe
# level + 3, levels >= 4 whole) into UFSM_HTTP_CACHE, whole shards per request (sample.c pfs_worker, ~90 MB/s vs ~20 per chunk).
# Logs to stdout: the cache size and free disk before / after, each source x level's tally (stored / absent / already cached / GB fetched,
# MB/s, coverage = share of the needed shards now cached), the totals. Exit 0 only when every shard came (coverage 1); 1 = fetch failures
# (the trainer would read those chunks cold: alert, relaunch the prefetch); 2 = usage / setup; 3 = less than MIN_FREE_GB free disk, at the
# start or at any shard during the fetch (ufsm prefetch --min-free-gb checks before each shard since 2026-11-25: the R11P prefetch, checked
# only at start, filled the pod disk to 9 GB free); what came stays cached, a rerun resumes.
# P: the trainer's window (--P, level voxels; default 256). Env: UFSM_HTTP_CACHE (required), U (ufsm binary, default /root/ufsm/build/ufsm),
# THREADS (24), LEVELS (9: the top level; a source without "max_level" is fetched up to it), MIN_FREE_GB (20: stop below it), DRY=1 (the needed chunks and the GB still to fetch per source x level, from the shard indexes; a
# pass of index reads, ~1-2 min per 1000 shards). The cache never evicts: size the sources (ct_zslabs, min/max_level) so their CT fits the
# disk. Every launch runs DRY=1 first and compares "GB to fetch" with the free disk (R11P: estimated ~60 GB from chunk counts, was > 150).
set -u
J=${1:?usage: ctprefetch.sh SOURCES.json [P]}; P=${2:-256}; U=${U:-/root/ufsm/build/ufsm}; C=${UFSM_HTTP_CACHE:?set UFSM_HTTP_CACHE (the trainer reads the same cache)}
[ -s "$J" ] || { echo "ctprefetch: no $J"; exit 2; }; [ -x "$U" ] || { echo "ctprefetch: no ufsm binary $U"; exit 2; }
mkdir -p "$C" || exit 2
say() { echo "$(date '+%F %T') ctprefetch: $*"; }
gb() { du -sb "$C" 2> /dev/null | awk '{ printf "%.1f", $1 / 1e9 }'; }
dfree() { df -B1G --output=avail "$C" | tail -1 | tr -d ' '; }
f0=$(dfree); [ "$f0" -ge "${MIN_FREE_GB:-20}" ] || { say "ALERT: $f0 GB free on the cache disk (< MIN_FREE_GB ${MIN_FREE_GB:-20}): not started"; exit 3; }
say "$J, window $P, cache $C ($(gb) GB, $f0 GB free), $U"
if [ "${DRY:-0}" = 1 ]; then CUDA_VISIBLE_DEVICES= "$U" prefetch "$J" --levels "${LEVELS:-9}" --P "$P" --threads "${THREADS:-24}" --dry 1 2>&1 | grep -E 'chunks needed|over all'; exit 0; fi
MF=${MIN_FREE_GB:-20}
t0=$(date +%s); LOG=$C/ctprefetch-$(date +%Y%m%d-%H%M%S).log
CUDA_VISIBLE_DEVICES= nice -n 5 "$U" prefetch "$J" --levels "${LEVELS:-9}" --P "$P" --threads "${THREADS:-24}" --min-free-gb "$MF" > "$LOG" 2>&1; rc=$?
tr '\r' '\n' < "$LOG" | grep -E 'level [0-9]+ done|over all|no occupancy|error|cannot|FAIL|ALERT' | sed 's/^/  /'
nf=$(grep -o '[0-9]* shard failures' "$LOG" | awk '{ s += $1 } END { print s + 0 }')
cov=$(grep -o 'coverage [0-9.]*' "$LOG" | awk '{ if ($2 < m || m == "") m = $2 } END { print m == "" ? "n/a" : m }')
say "done in $(( $(date +%s) - t0 )) s, rc $rc: cache $(gb) GB, $(dfree) GB free; $nf shard failures, lowest coverage $cov"
[ $rc = 0 ] && [ "$nf" = 0 ] && exit 0
[ $rc = 3 ] && { say "ALERT: stopped at $(dfree) GB free (< MIN_FREE_GB $MF): free disk or trim the sources, then rerun (resumes)"; exit 3; }
say "ALERT: prefetch incomplete (rc $rc, $nf shard failures): the trainer would read those chunks cold"; exit 1

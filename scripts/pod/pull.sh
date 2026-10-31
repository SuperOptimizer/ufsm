#!/usr/bin/env bash
# scripts/pod/pull.sh [options] USER@HOST:REMOTE_DIR LOCAL_DIR: pull a pod's output dir to the desk, resumable, then VERIFY it.
# Replaces the per-pod pull*.sh copies (10-31): one rsync recipe, retries, parallel streams, and a verify pass that must be clean before
# anyone says "SAFE TO KILL".
#   -p PORT        ssh port (22)            -i KEY      ssh identity file
#   -j N           parallel rsync streams (4): the top-level entries of REMOTE_DIR dealt round-robin by size
#   -x PATTERN     rsync --exclude (repeatable; e.g. -x fields -x 'slabs/*/f')
#   -c DIR         rsync --compare-dest DIR (files already there, e.g. an earlier tier on another disk, are not pulled again)
#   -t TRIES       attempts per stream (5; 30 s apart)
#   -v MODE        verify: size (default: names + sizes + mtimes) | sum (full checksums: reads every byte on both sides) | none
# Exit 0 only when the verify finds nothing left to transfer; the last line says VERIFIED (or lists what differs). Logs: LOCAL_DIR.pull.log.
# Never deletes anything on either side.
set -u
PORT=22; KEY=; J=4; TRIES=5; VERIFY=size; EXC=(); CMP=()
while getopts "p:i:j:x:c:t:v:" o; do case $o in
  p) PORT=$OPTARG;; i) KEY=$OPTARG;; j) J=$OPTARG;; x) EXC+=(--exclude "$OPTARG");; c) CMP+=(--compare-dest "$OPTARG");;
  t) TRIES=$OPTARG;; v) VERIFY=$OPTARG;; *) sed -n 2,15p "$0"; exit 2;; esac; done
shift $((OPTIND - 1)); SRC=${1:?usage: pull.sh [options] USER@HOST:REMOTE_DIR LOCAL_DIR}; DST=${2:?usage: pull.sh [options] USER@HOST:REMOTE_DIR LOCAL_DIR}
HOSTPART=${SRC%%:*}; RDIR=${SRC#*:}; RDIR=${RDIR%/}; DST=${DST%/}
SSH="ssh -p $PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ConnectTimeout=30${KEY:+ -i $KEY}"
LOG=$DST.pull.log; mkdir -p "$DST" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
RS=(rsync -a --partial --timeout=300 -e "$SSH" "${EXC[@]}" "${CMP[@]}")
# the top-level entries with their sizes (du on the pod), dealt to J lists, largest first
mapfile -t ENT < <($SSH $HOSTPART "cd '$RDIR' && ls -A | xargs -r -d '\\n' du -s --apparent-size --block-size=1 --" | sort -rn | awk -F'\t' '{print $2}')
[ ${#ENT[@]} -gt 0 ] || { log "pull: nothing listed in $SRC (ssh failed or empty)"; exit 1; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
for k in "${!ENT[@]}"; do echo "${ENT[$k]}" >> "$TMP/list$((k % J))"; done
log "pull: $SRC -> $DST: ${#ENT[@]} entries on $J streams (excludes: ${EXC[*]:-none})"
stream() {   # stream LISTFILE: rsync those entries, TRIES attempts
  local t rc; for t in $(seq 1 $TRIES); do
    "${RS[@]}" --files-from="$1" -r "$HOSTPART:$RDIR/" "$DST/" >> "$LOG.$(basename $1)" 2>&1; rc=$?
    [ $rc = 0 ] && return 0; [ $rc = 23 ] || [ $rc = 24 ] && return 0   # (vanished / partial attribute errors: the verify decides)
    echo "$(date '+%F %T') rsync rc $rc (try $t of $TRIES)" >> "$LOG.$(basename $1)"; sleep 30
  done; return 1
}
pids=(); for f in "$TMP"/list*; do stream "$f" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; cat "$LOG".list* >> "$LOG" 2> /dev/null; rm -f "$LOG".list*
[ $rc = 0 ] || log "pull: a stream gave up after $TRIES tries (verifying anyway)"
[ "$VERIFY" = none ] && { log "pull: done, NOT verified (-v none)"; exit $rc; }
# verify: a dry run must find nothing to transfer (size: sizes + mtimes; sum: checksums)
V=(); [ "$VERIFY" = sum ] && V=(--checksum)
DIFF=$("${RS[@]}" -n "${V[@]}" --out-format='%n' "$HOSTPART:$RDIR/" "$DST/" 2>&1 | grep -v '/$')
if [ -n "$DIFF" ]; then log "pull: NOT VERIFIED ($VERIFY): $(echo "$DIFF" | wc -l) entries differ, first: $(echo "$DIFF" | head -5 | tr '\n' ' ')"; exit 1; fi
NF=$(find "$DST" -type f | wc -l); SZ=$(du -sh --apparent-size "$DST" | cut -f1)
log "pull: VERIFIED ($VERIFY): $SRC == $DST ($NF files, $SZ$([ ${#EXC[@]} -gt 0 ] && echo ", excluding ${EXC[*]}"))"

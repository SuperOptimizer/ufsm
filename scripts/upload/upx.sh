#!/bin/bash
# upx.sh: fast, verified uploads to the dl.ash2txt SFTP account through up.sh (credentials stay in up.sh / ~/ash2txt;
# this script never reads them and all lftp output passes through up.sh's redaction).
#
#   upx.sh [-P N] [-g GLOB] [-r ROUNDS] [-n] [-l LOG] LOCAL REMOTE [LOCAL REMOTE ...]
#
#   LOCAL   a directory (mirrored recursively into REMOTE) or a file (put into the directory REMOTE)
#   REMOTE  path on the account, e.g. volcomp/PHercParis4/segments/ufsm-c2f-20261030/renders/wraps (created if absent)
#   -P N    parallel files per mirror (default auto: 8 when the mean file is >= 1 MB, 16 for smaller files)
#   -g GLOB only files matching GLOB (mirror --include-glob and the verify listing), e.g. -g '*.sfc'
#   -r N    rounds of upload + verify until every file is present with the local size (default 4)
#   -n      verify only (no upload)
#   -l LOG  append the (redacted) lftp output and the verify report here (default ~/reg-upload/upx.log)
#   UP=path the up.sh to use (default ~/reg-upload/up.sh)
#
# Measured 2026-10-30 (desk -> dl.ash2txt, memory ufsm-upload-speed-2026-10-30): the desk uplink (~4.9 MB/s total, also to
# Cloudflare) is the cap, not the server (48 parallel sessions accepted, single connection 4.1-4.3 MB/s). Big files: P 4-8
# reach 4.4-4.6 MB/s; small files cost ~0.17 s each serially (open/write/close/setstat round trips at 63 ms RTT), P 16 hides it.
# The server can NOT resume: reput / mirror --continue truncates the remote file to 0 bytes and fails ("Illegal seek"), so this
# never uses --continue; an interrupted file is re-sent whole by the next round (mirror compares sizes).
# Exit 0 = verified (every local file exists remotely with the same size), 1 = mismatches left after all rounds.
set -u
P=auto; G=; R=4; VONLY=0; LOG=$HOME/reg-upload/upx.log; UP=${UP:-$HOME/reg-upload/up.sh}
while getopts "P:g:r:nl:" o; do
  case $o in P) P=$OPTARG ;; g) G=$OPTARG ;; r) R=$OPTARG ;; n) VONLY=1 ;; l) LOG=$OPTARG ;; *) sed -n '2,15p' "$0"; exit 2 ;; esac
done
shift $((OPTIND - 1))
[ $# -ge 2 ] && [ $(($# % 2)) -eq 0 ] || { sed -n '2,15p' "$0"; exit 2; }
[ -x "$UP" ] || { echo "upx: $UP not executable" >&2; exit 2; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# local listing "<relpath>\t<size>" for pair i (a file pair lists just the basename)
llist() {
  local L=$1
  if [ -d "$L" ]; then
    if [ -n "$G" ]; then find "$L" -type f -name "$G" -printf '%P\t%s\n'; else find "$L" -type f -printf '%P\t%s\n'; fi
  else
    printf '%s\t%s\n' "$(basename "$L")" "$(stat -c %s "$L")"
  fi | LC_ALL=C sort
}

# lftp settings: more patient reconnects than up.sh's 3 retries, no chmod round trip per file
SETS='set net:max-retries 10
set net:reconnect-interval-base 5
set net:reconnect-interval-multiplier 2
set net:reconnect-interval-max 120
set net:timeout 60
set mirror:set-permissions no
set xfer:clobber yes'

pairs=("$@"); np=$(($# / 2))
bytes=0; files=0
for ((i = 0; i < np; i++)); do
  L=${pairs[2 * i]}
  [ -e "$L" ] || { echo "upx: missing local $L" >&2; exit 2; }
  llist "$L" > "$T/l$i"
  bytes=$((bytes + $(awk -F'\t' '{s += $2} END {print s + 0}' "$T/l$i"))); files=$((files + $(wc -l < "$T/l$i")))
done
if [ "$P" = auto ]; then P=8; [ "$files" -gt 0 ] && [ $((bytes / files)) -lt 1000000 ] && P=16; fi
log "upx: $np pair(s), $files files, $((bytes / 1000000)) MB, parallel $P, rounds $R"

# remote listing of every pair in one session -> $T/r<i> as "<relpath>\t<size>"
verify() {
  local i R0 bad=0
  PEND=0
  {
    echo "$SETS"
    for ((i = 0; i < np; i++)); do
      R0=${pairs[2 * i + 1]}
      if [ -d "${pairs[2 * i]}" ]; then echo "find -l \"$R0\""; else echo "find -l \"$R0/$(basename "${pairs[2 * i]}")\""; fi
    done
  } | "$UP" > "$T/rall" 2>&1
  # find prints paths with the remote prefix: select each pair's lines by it (no markers: lftp buffers its stdout)
  for ((i = 0; i < np; i++)); do
    R0=${pairs[2 * i + 1]}
    awk -v pre="$R0/" '
      /^-/ { line = $0; size = $3; sub(/^[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ /, "", line)
             if (index(line, pre) == 1) print substr(line, length(pre) + 1) "\t" size }' "$T/rall" |
      LC_ALL=C sort > "$T/r$i"
    # local files missing remotely or with another size
    LC_ALL=C join -t $'\t' -a 1 -e MISSING -o 0,1.2,2.2 "$T/l$i" "$T/r$i" | awk -F'\t' '$2 != $3' > "$T/bad$i"
    OK[i]=1
    if [ -s "$T/bad$i" ]; then
      n=$(wc -l < "$T/bad$i"); bad=$((bad + n)); OK[i]=0
      PEND=$((PEND + $(awk -F"\t" "{s += \$2} END {print s + 0}" "$T/bad$i")))
      log "verify ${pairs[2 * i + 1]}: $n file(s) missing or wrong size (path local remote):"
      head -n 5 "$T/bad$i" | sed 's/^/    /' | tee -a "$LOG"
    else
      log "verify ${pairs[2 * i + 1]}: $(wc -l < "$T/l$i") files OK"
    fi
  done
  return $((bad > 0))
}
if [ "$VONLY" = 1 ]; then verify; exit $?; fi
verify && { log "upx: DONE (already complete)"; exit 0; }
for ((round = 1; round <= R; round++)); do
  s=$(date +%s); pend=$PEND
  {
    echo "$SETS"
    for ((i = 0; i < np; i++)); do
      [ "${OK[i]}" = 1 ] && continue
      L=${pairs[2 * i]}; R0=${pairs[2 * i + 1]}
      echo "mkdir -p -f \"$R0\""
      if [ -d "$L" ]; then
        echo "mirror -R --parallel=$P --no-perms ${G:+--include-glob \"$G\"} \"$L\" \"$R0\""
      else
        echo "put -O \"$R0\" \"$L\""
      fi
    done
  } | "$UP" >> "$LOG" 2>&1
  e=$(date +%s)
  log "round $round: $((pend / 1000000)) MB pending, $((e - s)) s, $(awk -v b=$pend -v t=$((e - s)) 'BEGIN {printf "%.2f", (t > 0 ? b / t / 1e6 : 0)}') MB/s"
  if verify; then log "upx: DONE verified"; exit 0; fi
done
log "upx: FAILED, mismatches left after $R rounds"
exit 1

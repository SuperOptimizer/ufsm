#!/usr/bin/env bash
# scripts/inksmoke/gtfetch.sh MANIFEST.tsv DEST: the human ink GT from its public upstream (gtmanifest.py rows: relpath url size md5),
# 8 parallel downloads, resumable; every file's size + md5 checked (a mismatch is deleted and reported). Exit 0 only when all files check.
set -uo pipefail; M=${1:?manifest}; D=${2:?dest}; mkdir -p "$D"
one() { local rel=$1 url=$2 sz=$3 m=$4 p="$D/$1"
  if [ -s "$p" ] && [ "$(stat -c %s "$p")" = "$sz" ] && [ "$(md5sum < "$p" | cut -d' ' -f1)" = "$m" ]; then return 0; fi
  mkdir -p "$(dirname "$p")"
  curl -sfL --retry 5 --retry-delay 5 --max-time 3600 -o "$p.part" "$url" && mv "$p.part" "$p" || { rm -f "$p.part"; echo "FAIL get $rel"; return 1; }
  [ "$(stat -c %s "$p")" = "$sz" ] && [ "$(md5sum < "$p" | cut -d' ' -f1)" = "$m" ] || { rm -f "$p"; echo "FAIL check $rel"; return 1; }; }
export -f one; export D
tr '\t' ' ' < "$M" | xargs -P 8 -L 1 bash -c 'one "$@"' _ | tee "$D/.gtfetch.err"
n=$(grep -c FAIL "$D/.gtfetch.err"); echo "gtfetch: $(wc -l < "$M") files, $n failed"; [ "$n" = 0 ]

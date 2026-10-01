#!/bin/sh
# Parallel ranged download with resume: pget.sh <url> <out> [nconn] [bearer-token-file]
# Splits the object into nconn byte ranges, downloads each into <out>.partN with curl -C -, then concatenates.
set -e
URL=$1; OUT=$2; N=${3:-16}; TOKF=$4
AUTH=""; [ -n "$TOKF" ] && AUTH="Authorization: Bearer $(cat "$TOKF")"
SIZE=$(curl -sIL ${AUTH:+-H "$AUTH"} "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
[ -n "$SIZE" ] || { echo "no content-length" >&2; exit 1; }
FINAL=$(curl -s -o /dev/null -w '%{redirect_url}' ${AUTH:+-H "$AUTH"} "$URL"); [ -n "$FINAL" ] || FINAL=$URL
echo "size $SIZE bytes, $N connections, from ${FINAL%%\?*}" >&2
SEG=$(( (SIZE + N - 1) / N ))
i=0
while [ $i -lt $N ]; do
  A=$((i * SEG)); B=$(( (i + 1) * SEG - 1 )); [ $B -ge $SIZE ] && B=$((SIZE - 1))
  P="$OUT.part$i"
  HAVE=0; [ -f "$P" ] && HAVE=$(stat -c %s "$P")
  if [ $HAVE -lt $((B - A + 1)) ]; then
    ( while :; do
        HAVE=0; [ -f "$P" ] && HAVE=$(stat -c %s "$P")
        [ $HAVE -ge $((B - A + 1)) ] && break
        curl -sS -L -r $((A + HAVE))-$B ${AUTH:+-H "$AUTH"} "$FINAL" >> "$P" 2>>"$OUT.log" && break
        sleep 5
      done ) &
  fi
  i=$((i + 1))
done
wait
i=0; : > "$OUT.tmp"
while [ $i -lt $N ]; do cat "$OUT.part$i" >> "$OUT.tmp"; i=$((i + 1)); done
[ "$(stat -c %s "$OUT.tmp")" = "$SIZE" ] || { echo "size mismatch" >&2; exit 1; }
mv "$OUT.tmp" "$OUT"; i=0; while [ $i -lt $N ]; do rm -f "$OUT.part$i"; i=$((i + 1)); done
echo "done: $OUT" >&2

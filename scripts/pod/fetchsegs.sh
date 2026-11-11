#!/usr/bin/env bash
# fetchsegs.sh SCROLL DEST [GREP-E]: the published segment meshes (meta.json + x/y/z.tif of mesh/*-on-<scan>-*.tifxyz) of one scroll from the
# public open-data bucket, 16 parallel downloads, resumable. Paris 4: June (20260623*-w*, no jordi) + July (20260701*) on the 2.4 um scan
# (as scripts/wbench/gtjuly/fetch.sh and rv/sense lib.sh p4june / p4july).
set -u; S=$1; D=$2; F=${3:-.}; B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com; mkdir -p $D
segs=$(curl -s --max-time 120 "$B/?list-type=2&prefix=$S/segments/&delimiter=/&max-keys=1000" | grep -o '<Prefix>[^<]*</Prefix>' | sed 's/<[^>]*>//g' | grep -v "^$S/segments/\$" | grep -E -- "$F" || true)
[ $S = PHercParis4 ] && segs=$(echo "$segs" | grep -E '/(20260623[0-9]*-w[0-9]|20260701)' | grep -v jordi)
for sp in $segs; do curl -s --max-time 120 "$B/?list-type=2&prefix=${sp}mesh/&max-keys=1000" | grep -o '<Key>[^<]*</Key>' | sed 's/<[^>]*>//g' \
  | grep -v intermediate | grep -E '/(meta\.json|[xyz]\.tif)$'; done > $D/keys.txt
echo "$S: $(echo "$segs" | grep -c .) segments, $(wc -l < $D/keys.txt) files"
xargs -P 16 -I{} sh -c 'f="$1"; d="$2/$(echo "$f" | sed "s|^[^/]*/segments/||")"; mkdir -p "$(dirname "$d")"; [ -s "$d" ] || curl -sf --retry 5 --max-time 3600 -o "$d.part" "$3/$f" && mv "$d.part" "$d"' _ {} $D $B < $D/keys.txt
touch $D/.done; du -sh $D

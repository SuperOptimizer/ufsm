#!/bin/sh
# Export every tifxyz segment in the open-data bucket to surfcomp: gt/segments/<scroll>/<segment-id>.<volume>.sfc
set -u
B=/home/forrest/ufsm/build/ufsm
S3=https://vesuvius-challenge-open-data.s3.amazonaws.com
while read -r scroll seg vol path; do
  out=/vesuvius/ufsm/gt/segments/$scroll/$seg.$vol.sfc
  [ -f "$out" ] && continue
  mkdir -p "$(dirname "$out")"
  $B ingest-mesh "$S3" "$path" "$out" || echo "FAILED $scroll $seg $path"
done < /vesuvius/ufsm/segments.txt
echo "segments done"

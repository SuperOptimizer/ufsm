#!/bin/bash
# field_paste_link.sh ORIG BOXED OUT: a fit-field set OUT that is ORIG (zarr, 128-cell shards) with the shards of BOXED (field-blend
# --paste F --box z,y,x,nz,ny,nx ORIG ..., written on ORIG's shard grid) in place of ORIG's: symlinks, no copies. OUT's fields.json
# is ORIG's (the axis table, origin and shape stay whole) plus a "pasted" key naming BOXED. Planes BOXED lacks stay ORIG's.
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: field_paste_link.sh ORIG BOXED OUT" >&2; exit 2; }
O=$(realpath "$1"); B=$(realpath "$2"); OUT=$3
[ -e "$OUT" ] && { echo "field_paste_link: $OUT exists" >&2; exit 1; }
read -r BZ BY BX < <(python3 -c 'import json,sys; b=json.load(open(sys.argv[1]))["blend"]["box"]; print(b[0]//128, b[1]//128, b[2]//128)' "$B/fields.json")
mkdir -p "$OUT"
python3 - "$O/fields.json" "$OUT/fields.json" "$B" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); j["pasted"] = sys.argv[3]; json.dump(j, open(sys.argv[2], "w"))
EOF
n=0
for pd in "$O"/*/; do
    p=$(basename "$pd"); [ -f "$pd/zarr.json" ] || continue
    if [ ! -f "$B/$p/zarr.json" ]; then ln -s "$O/$p" "$OUT/$p"; continue; fi
    cp -rs "$O/$p" "$OUT/$p"   # (symlinks to every shard and a linked zarr.json)
    (cd "$B/$p" && find c -type f) | while read -r f; do   # c/z/y/x in BOXED -> c/(z+BZ)/(y+BY)/(x+BX)
        IFS=/ read -r _ z y x <<< "$f"; t="c/$((z + BZ))/$((y + BY))/$((x + BX))"
        mkdir -p "$OUT/$p/$(dirname "$t")"; ln -sfn "$B/$p/$f" "$OUT/$p/$t"
    done
    n=$((n + 1))
done
echo "field_paste_link: $n planes overlaid with $B (shard offset $BZ $BY $BX) -> $OUT"

#!/bin/sh
# Download the published surfcomp segments (every segment registered on every scan of its scroll) and
# rasterize them per (scroll, volume) into label pyramids under /vesuvius/ufsm/gt/raster/<scroll>/<vol>/.
# Level: 0 for >= 5 um scans, 1 for 1.5-5 um, 2 below.
set -u
B=/home/forrest/ufsm/build/ufsm
SC=https://dl.ash2txt.org/community-uploads/forrest/surfcomp
VC=https://dl.ash2txt.org/community-uploads/forrest/volcomp
MAN=/vesuvius/usrm/surfcomp-manifest.json
DL=/vesuvius/ufsm/gt/surfcomp
OUT=/vesuvius/ufsm/gt/raster
T=${T:-3}
python3 - "$MAN" > /vesuvius/ufsm/surfcomp.txt <<'PY'
import json, sys, re
m = json.load(open(sys.argv[1]))
for s in m["surfaces"]:
    r = re.search(r"-on-(\d+)-([0-9.]+)um$", s["name"])
    if r: print(s["scroll"], r.group(1), r.group(2), s["path"])
PY
# download
while read -r scroll vol um path; do
  f="$DL/$path"; [ -f "$f" ] && continue
  mkdir -p "$(dirname "$f")"
  curl -sf -o "$f.tmp" "$SC/$path" && mv "$f.tmp" "$f" || echo "download failed $path"
done < /vesuvius/ufsm/surfcomp.txt
echo "downloads done: $(find $DL -name '*.sfc' | wc -l) files"
# rasterize per (scroll, vol)
awk '{print $1, $2, $3}' /vesuvius/ufsm/surfcomp.txt | sort -u | while read -r scroll vol um; do
  dir="$OUT/$scroll/$vol"
  if [ -f "$dir/zarr.json" ]; then echo "skip $scroll $vol (done)"; continue; fi
  key=$(curl -s "$VC/$scroll/volumes/" | grep -o "href=\"$vol-[^\"]*\.zarr/\"" | sed 's/href="//;s/\/"//' | head -1)
  [ -n "$key" ] || { echo "skip $scroll $vol (no volcomp mirror)"; continue; }
  shape=$($B info "$VC" "$scroll/volumes/$key" 2>/dev/null | sed -n 2p | sed 's/.*shape \([0-9]*\) x \([0-9]*\) x \([0-9]*\).*/\1,\2,\3/')
  [ -n "$shape" ] || { echo "skip $scroll $vol (no info)"; continue; }
  level=0; awk "BEGIN{exit !($um < 5)}" && level=1; awk "BEGIN{exit !($um < 1.5)}" && level=2
  meshes=$(awk -v s="$scroll" -v v="$vol" '$1==s && $2==v {print "'"$DL"'/" $4}' /vesuvius/ufsm/surfcomp.txt)
  echo "== $scroll $vol: $(echo "$meshes" | wc -w) meshes, $um um, shape $shape, level $level"
  $B raster "$dir" --shape "$shape" --um "$um" --level $level --T $T --levels 5 --threads 8 $meshes || echo "FAILED $scroll $vol"
done
echo "surfcomp raster done"

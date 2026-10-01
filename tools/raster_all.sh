#!/bin/sh
# Rasterize exported segment meshes into label pyramids, one per (scroll, original volume), for the
# volumes that have a volcomp CT mirror and no HF label zarr. Output: /vesuvius/ufsm/gt/raster/<scroll>/<vol>/
set -u
B=/home/forrest/ufsm/build/ufsm
ROOT=https://dl.ash2txt.org/community-uploads/forrest/volcomp
OUT=/vesuvius/ufsm/gt/raster
SEG=/vesuvius/ufsm/segments.txt
T=${T:-3}
# scrolls whose 2 um labels already come from the HF zarrs
SKIP="PHerc0500P2 PHerc0343P PHercMANBp PHerc1667"
awk '{print $1, $3}' "$SEG" | sort -u | while read -r scroll vol; do
  case " $SKIP " in *" $scroll "*) echo "skip $scroll $vol (HF labels)"; continue;; esac
  key=$(curl -s "$ROOT/$scroll/volumes/" | grep -o "href=\"$vol-[^\"]*\.zarr/\"" | sed 's/href="//;s/\/"//' | head -1)
  if [ -z "$key" ]; then echo "skip $scroll $vol (no volcomp mirror)"; continue; fi
  dir="$OUT/$scroll/$vol"
  if [ -f "$dir/zarr.json" ]; then echo "skip $scroll $vol (done)"; continue; fi
  info=$($B info "$ROOT" "$scroll/volumes/$key" 2>/dev/null | sed -n 2p)
  um=$(echo "$key" | sed -n 's/.*-\([0-9.]*\)um-.*/\1/p'); shape=$(echo "$info" | sed 's/.*shape \([0-9]*\) x \([0-9]*\) x \([0-9]*\).*/\1,\2,\3/')
  [ -n "$um" ] || { echo "skip $scroll $vol (no info)"; continue; }
  level=0; awk "BEGIN{exit !($um < 5)}" && level=1
  meshes=$(ls /vesuvius/ufsm/gt/segments/$scroll/*.$vol.sfc 2>/dev/null)
  n=$(echo "$meshes" | wc -w)
  echo "== $scroll $vol: $n meshes, $um um, shape $shape, level $level"
  $B raster "$dir" --shape "$shape" --um "$um" --level $level --T $T --levels 5 --threads 8 $meshes || echo "FAILED $scroll $vol"
done
echo "raster done"

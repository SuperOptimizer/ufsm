#!/usr/bin/env bash
# scripts/u0/inventory.sh: Phase U0 inventory of every ESRF scan on the volcomp mirror (no CT downloads: HTTP listings + zarr.json only).
# Per volume: the level nearest 9.6 um (native 8.64 / 9.36 um at level 0, 2.2-2.4 um at level 2, 1.13 um at 3, 0.55 um at 4), its shape,
# the compressed size of that level and of the coarse levels the copy also takes (L+2 and coarser: scanprep's mask grid and the sampler's
# coarse probe at L+3), the shard file count, and whether a local copy already exists under /vesuvius/ufsm/volcomp.
# usage: inventory.sh OUT.tsv [JOBS]     (bash + curl + jq + awk; run on the desk: the laptop's network is slow)
set -u
OUT=${1:?out.tsv}; J=${2:-16}
R=${R:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
OLD=${OLD:-/vesuvius/ufsm/volcomp}
REG=${REG:-}   # optional: a file of "pairname coarse-id fine-id" lines (registration pairs) for the pair column
export R OLD REG
lst() { curl -sf --retry 3 "$1" | grep -o 'href="[^"]*"' | sed 's/href="//;s/"$//' | grep -v '^\.\./\?$'; }
# the listing of one directory as "name bytes" lines (sizes as the index prints them: B / KiB / MiB / GiB)
lsz() { curl -sf --retry 3 "$1" | grep -o 'href="[^"]*"[^<]*</a></td><td class="size">[^<]*' |
        sed 's/href="\([^"]*\)".*size">/\1 /' | awk '{v=$2; u=$3; m=1; if(u=="KiB")m=1024; else if(u=="MiB")m=1048576; else if(u=="GiB")m=1073741824;
                                                   if(v=="-") print $1, -1; else printf "%s %.0f\n", $1, v*m}'; }
# bytes and files of one level directory (shard files under c/z/y/x)
level_bytes() {
  local L=$1 tot=0 nf=0 z y
  for z in $(lst "$L/c/" | grep /$); do
    for y in $(lst "$L/c/$z" | grep /$); do
      read -r b f < <(lsz "$L/c/$z$y" | awk '$2>=0{b+=$2; f++} END{printf "%.0f %d\n", b, f}')
      tot=$((tot + b)); nf=$((nf + f))
    done
  done
  echo "$tot $nf"
}
export -f lst lsz level_bytes
one() {
  local key=${1%/}; local s=${key%%/*} v=${key##*/} um kev dist id lev nlev L shape bytes files cbytes cl
  v=${v%/}; id=${v%%-*}
  um=$(echo "$v" | grep -o '[0-9.]*um' | head -1 | tr -d um)
  kev=$(echo "$v" | grep -o '[0-9]*keV' | head -1 | tr -d keV); dist=$(echo "$v" | grep -o -- '-[0-9.]*m-' | head -1 | tr -d m-)
  local G; G=$(curl -sf --retry 3 "$R/$key/zarr.json") || { printf '%s\t%s\t%s\t%s\t%s\t%s\tERR\n' "$s" "$id" "$v" "$um" "${kev:--}" "${dist:--}"; return; }
  nlev=$(echo "$G" | jq '.attributes.ome.multiscales[0].datasets | length')
  # the level whose voxel size um * 2^L is nearest 9.6 um (in log)
  lev=$(awk -v u="$um" -v n="$nlev" 'BEGIN{b=0; bd=1e9; for(l=0;l<n;l++){d=log(u*2^l/9.6); if(d<0)d=-d; if(d<bd){bd=d; b=l}} print b}')
  L="$R/$key/$lev"
  shape=$(curl -sf --retry 3 "$L/zarr.json" | jq -c '.shape' | tr -d '[] ')
  read -r bytes files < <(level_bytes "$L")
  cbytes=0
  for ((cl = lev + 2; cl < nlev; cl++)); do read -r b f < <(level_bytes "$R/$key/$cl"); cbytes=$((cbytes + b)); done
  local loc=-; [ -d "$OLD/$s/volumes/$v/$lev/c" ] && loc=$(find "$OLD/$s/volumes/$v/$lev/c" -type f | wc -l)
  [ "$loc" = - ] && [ -d "$OLD/$s/$v/$lev/c" ] && loc="cache:$(find "$OLD/$s/$v/$lev/c" -type f | wc -l)"
  local pair=-; [ -n "$REG" ] && [ -f "$REG" ] && pair=$(awk -v i="$id" '$2==i||$3==i{printf "%s%s", (n++?",":""), $1} END{if(!n)printf "-"}' "$REG")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.4f\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$id" "$v" "$um" "${kev:--}" "${dist:--}" "$lev" "$nlev" \
    "$(awk -v u="$um" -v l="$lev" 'BEGIN{print u*2^l}')" "$shape" "$bytes" "$files" "$cbytes" "$loc" "$pair"
}
export -f one
{
  printf 'scroll\tid\tvolume\tum\tkeV\tdist_m\tlevel\tnlev\tlevel_um\tshape_zyx\tlevel_bytes\tshards\tcoarse_bytes\tlocal\tpairs\n'
  for s in $(lst "$R/" | grep /$); do for v in $(lst "$R/${s}volumes/" | grep zarr/$); do echo "${s}volumes/$v"; done; done |
    xargs -P "$J" -I{} bash -c 'one "$@"' _ {} | sort
} > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
echo "inventory: $(($(wc -l < "$OUT") - 1)) volumes -> $OUT"

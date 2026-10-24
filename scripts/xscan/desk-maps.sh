#!/usr/bin/env bash
# desk-maps.sh [SRC_HOST] [OUT]: --aug-xscan partner maps for the desk with local roots.
# Copies the Paris 4 ufsm reg maps (reg.json + field.f32) from the laptop (default forrest@192.168.0.175:~/ufsm-data/reg/pairs)
# into OUT (default /vesuvius/ufsm/xscan/<pair>/) and rewrites the HTTP mirror roots to the desk copies:
#   coarse 78 keV 2.4 um  /vesuvius/ufsm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr (the training source)
#   fine   137 keV 2.4 um /vesuvius/ufsm/u0/vol/PHercParis4/20260323153942-2.400um-0.2m-137keV-masked.zarr (level 2 = 9.6 um)
#   fine   1.1 um 78 keV  /vesuvius/ufsm/u0/vol/PHercParis4/20260608103018-1.129um-0.2m-78keV-masked.zarr (level 3 = 9.03 um)
# Then add to the Paris 4 78 keV source in sources.json:  "xscan": [{"map": "OUT/PParis4_2to2/reg.json"}, {"map": "OUT/PParis4_2to1/reg.json"}]
set -e
SRC=${1:-forrest@192.168.0.175:/home/forrest/ufsm-data/reg/pairs}
OUT=${2:-/vesuvius/ufsm/xscan}
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
for p in PParis4_2to2 PParis4_2to1; do
    mkdir -p "$OUT/$p"
    case $SRC in *:*) scp -q "$SRC/$p/final/reg.json" "$SRC/$p/final/field.f32" "$OUT/$p/";; *) cp "$SRC/$p/final/reg.json" "$SRC/$p/final/field.f32" "$OUT/$p/";; esac
    sed -i -e "s#\"root\":\"$M\",\"key\":\"PHercParis4/volumes/20260411134726#\"root\":\"/vesuvius/ufsm/volcomp\",\"key\":\"PHercParis4/20260411134726#" \
           -e "s#\"root\":\"$M\",\"key\":\"PHercParis4/volumes/#\"root\":\"/vesuvius/ufsm/u0/vol\",\"key\":\"PHercParis4/#" "$OUT/$p/reg.json"
    grep -o '"root":"[^"]*","key":"[^"]*"' "$OUT/$p/reg.json"
done

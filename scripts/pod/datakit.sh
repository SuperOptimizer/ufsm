#!/usr/bin/env bash
# datakit.sh K [STEP...]: make a fresh pod self-sufficient for the latent winding runs and the winding bench WITHOUT uploading label data
# (the desk uplink is ~4-5 MB/s; dl.ash2txt.org / the open-data bucket download at 20+ MB/s). Steps (default all, each restartable):
#   segs    published segment meshes of Paris 4 (June / July), PHerc0139, PHerc0500P2, PHerc1667 from the open-data bucket (fetchsegs.sh)
#   p4      Paris 4 codes A / B, phase A / B and the wbench q-9p6 codes (p4kit.sh; the 10-25 rebuild of this chain matched the desk)
#   0500    PHerc0500P2 at native 9.362 um (scripts/cal/gt.sh NATIVE=1)
#   0139    PHerc0139 at native 9.362 um from the q-assigned surfaces qA / qB (k0139.sh; K/gt/p0139-q is the one label input copied
#           from the desk, 55 MB, because its 10-16 wind-seed predates today's sense rule)
#   links   K/wbgt layout for scripts/wbench/site-pod.env, sources json K/sources-wind-gt-pod.json (mksources.sh)
# Small inputs copied from the desk once into K/small (205 MB): axis / umbilicus JSONs, the KLT basis klt1.lcb, wind96, boxes.tsv,
# holdouts.json. CT is read over HTTP from the volcomp mirror (byte-identical to the desk's v2/ct shards) through K/cache.
# Env U (ufsm binary), T (threads). Tools: gcc-14, pkg-config, libblosc/xxhash/zstd/lz4/curl/sqlite3 dev, bc, jq (see DATAKIT.md).
set -uo pipefail; K=${1:?K}; shift; D=$(dirname "$(readlink -f "$0")"); U=${U:-/workspace/lat/build/ufsm}; T=${T:-10}
ST=${*:-segs p4 0500 0139 links}; mkdir -p $K/seg $K/gt $K/cache
for s in $ST; do echo "== $(date +%T) datakit $s"; case $s in
  segs) for S in PHercParis4 PHerc0139 PHerc0500P2 PHerc1667; do [ -e $K/seg/$S/.done ] || bash $D/fetchsegs.sh $S $K/seg/$S & done; wait;;
  p4) U=$U T=$T bash $D/p4kit.sh $K;;
  0500) (cd $K && C=$K X=$K/small/gtaxis U=$U T=$T NATIVE=1 bash $D/../cal/gt.sh PHerc0500P2);;
  0139) U=$U T=$T bash $D/k0139.sh $K;;
  links) mkdir -p $K/wbgt/paris4-q9p6 $K/wbgt/paris4-wind-20261019 $K/wb/res $K/wb/fields
    ln -sfn $K/gt/paris4/q-9p6.zarr $K/wbgt/paris4-q9p6/q-9p6.zarr; ln -sfn $K/gt/paris4/phaseA $K/wbgt/paris4-wind-20261019/phaseA
    ln -sfn $K/gt/paris4/phaseB $K/wbgt/paris4-wind-20261019/phaseB; cp -n $K/small/boxes.tsv $K/small/holdouts.json $K/wb/
    bash $D/mksources.sh $K $K/sources-wind-gt-pod.json;;
  *) echo "unknown step $s"; exit 2;; esac; done
echo "== $(date +%T) datakit done"

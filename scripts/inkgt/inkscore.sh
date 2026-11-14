#!/bin/bash
# inkscore.sh TAG STRIPES-DIR [MAP-DIR]: scripts/inkgt/wrapscore.py on a stripe set against the published human ink labels -> OUTD/inkscore-TAG.{tsv,log}.
# CPU only. Reports and gates on the HELD-OUT samples only (validation rasters + the scan's holdout boxes; audit 11-14 #4: the "all" AUC is
# ~76 % training labels on 1667). No MAP-DIR: surface coverage only.
#   env: SCROLL (PHerc1667), SEGS (inkgt/segs-SCROLL.json), CTKEY (the labels' CT key), OUTD (/vesuvius/ufsm/unroll/sched), STEP (8),
#        GATE_AUC (0 = off: exit 3 when the held-out ink AUC at 40 um is below it), MEM (24G)
set -u
TAG=${1:?tag}; ST=${2:?stripes dir}; MP=${3:-/nonexistent}; SCROLL=${SCROLL:-PHerc1667}; OUTD=${OUTD:-/vesuvius/ufsm/unroll/sched}
SEGS=${SEGS:-/vesuvius/ufsm/inkgt/segs-$SCROLL.json}; PY=$(dirname "$(readlink -f "$0")")/wrapscore.py
case $SCROLL in
  PHerc1667) CK=PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr ;;
  PHerc1299) CK=PHerc1299/20260309130042-2.399um-0.2m-78keV-masked.zarr ;;
  PHercParis4) CK=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr ;;
  *) CK= ;;
esac
CTKEY=${CTKEY:-$CK}; [ -n "$CTKEY" ] || { echo "inkscore: CTKEY for $SCROLL"; exit 2; }
[ -d "$ST" ] || { echo "inkscore: no stripes $ST"; exit 1; }; [ -s "$SEGS" ] || { echo "inkscore: no labels $SEGS"; exit 1; }
[ "$MP" = /nonexistent ] || [ -d "$MP" ] || { echo "inkscore: no renders $MP"; exit 1; }
CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m ${MEM:-24G} python3 -I "$PY" "$ST" "$SEGS" $OUTD/inkscore-$TAG.tsv --map "$MP" --ct-key $CTKEY \
  --step ${STEP:-8} --tol 20,40,80 --gate-auc ${GATE_AUC:-0} > $OUTD/inkscore-$TAG.log 2>&1
rc=$?; echo "inkscore $TAG rc $rc"
grep -E "^miss|no .* map|not proportional|^HEADLINE|^WARNING|^GATE" $OUTD/inkscore-$TAG.log | head -6
awk -F'\t' 'NR == 1 || $2 == "heldout" || $2 == "rest"' $OUTD/inkscore-$TAG.tsv 2> /dev/null
exit $rc

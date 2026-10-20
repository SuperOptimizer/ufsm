#!/usr/bin/env bash
# ink.sh S: 3D ink of scroll S's 8-9 um scan on the desk with ink96-wk1 (Paris4 + Paris3, 9.6 um) at the scan's native spacing
# (--um = the scan's voxel size, level 0). User rules: no TTA, no overlap (--halo 0), the largest full-speed window, --shard = window,
# --occupancy 5 (air shards skipped), --full-grid 1 (one zarr on the CT's whole grid) --resume 1, retry loop.
# Phase 1: the fitted box regions (A/<S>/surf/boxes.tsv from fit.sh), phase 2: the whole scroll + the pyramid.
# Output A/<S>/ink/<scan>-ink96.zarr. Stops (exit 3) when /vesuvius has less than MINFREE GB (default 150) free.
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
s=$1; read -r scan um fz <<< "$(scan_of $s)"; [ -n "${fz:-}" ] || { echo "ink.sh: $s not in $A/scrolls.tsv"; exit 2; }
S=$A/$s; mkdir -p $S/ink $S/state; OUT=$S/ink/${scan%-masked}-ink96.zarr; K=$s/volumes/$scan.zarr
CKPT=${CKPT:-$A/bin/ink96-wk1-best.ckpt}; W=${W:-1024}; GPUS=${GPUS:-0,1}
space() { local f=$(df -BG --output=avail /vesuvius | tail -1 | tr -dc 0-9); [ $f -ge ${MINFREE:-150} ] || { echo "ink.sh: only ${f} GB free on /vesuvius, stopping"; exit 3; }; }
P() { $U predict $CKPT $R $K $OUT --um $um --level 0 --shard $W --window $W --halo 0 --ema 1 --q 32 --threads 32 --gpus $GPUS \
        --full-grid 1 --resume 1 --occupancy 5 "$@"; }
retry() { local n=0; until P "$@" >> $S/ink/predict.log 2>&1; do n=$((n + 1)); [ $n -ge 5 ] && return 1; space; log "$s ink predict failed ($n), retrying"; sleep 60; done; }
space
read -r Z Y X <<< "$($U info $R $K 2>/dev/null | awk '$1 == "0" {print $5, $7, $9}')"
[ -n "${X:-}" ] || { echo "ink.sh: cannot read the shape of $K"; exit 1; }
if [ -s $S/surf/boxes.tsv ] && ! done_ $s ink-boxes; then   # name z y x n (field cells, h 2)
  while read -r b z y x n; do
    [ -e $S/ink/$b.done ] && continue; space; t0=$(date +%s)
    retry --box $((2 * z)),$((2 * y)),$((2 * x)),$((2 * n)),$((2 * n)),$((2 * n)) --levels 1 || exit 1
    echo "$(date '+%F %T') $(( $(date +%s) - t0 ))s" > $S/ink/$b.done
  done < $S/surf/boxes.tsv
  mark $s ink-boxes
fi
space; t0=$(date +%s)
retry --box 0,0,0,$Z,$Y,$X --levels 6 || exit 1
mark $s ink "$(( $(date +%s) - t0 ))s whole, $(du -sh $OUT | cut -f1)"

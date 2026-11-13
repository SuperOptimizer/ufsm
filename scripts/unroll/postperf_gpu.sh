#!/usr/bin/env bash
# postperf_gpu.sh [OUT]: the GPU-window checks of postperf-20261112 (the post-slab chain speed-ups), ~5 min on one GPU of the desk, peak host RAM ~35 GB (at most 39):
#   1. stripimg's stretch renders: the Paris 4 u1 run's 3 middle stripes (ws010..ws012) rendered one after another (the old schedule)
#      and at once (unroll.sh SJOBS=3), wall times of both; every output file of the two must be identical, and each fill map equal to
#      the run's strip/fillmap/<s>_fill.tif; strip-mosaic --stretch-only on them must print the run's stretch (strip/stretch.txt)
#   2. wrap-snap's chunk decode on the GPU (unroll.sh SNAPGPU) against the CPU's: wall times, the .sfc identical to the run's
#   3. test_fit_join (tests/suite.txt, tag gpu: fit-join's recycled field blocks / sampling run there too)
# Reads the run read-only (P), writes only under OUT. Env: U (ufsm of the branch), P (the run), GPU (CUDA device, 0), RMEM (UFSM_RENDER_MEM
# GB per render, 7: a chunk-cache budget, the same images; the run's stretch renders were 19.6-20.9 GB RSS at 16, i.e. ~4.5 GB + the budget,
# so ~11.5 GB each and ~35 GB for the three at once), RCAP (memcap per render, 13G: the three at once can never pass 39 GB).
# Peak host RAM: the three renders at once (~35 GB, hard cap 3 x RCAP); the snaps ~7 GB, test_fit_join's build / run <= 16 GB, one at a time.
set -u
U=${U:-$HOME/ufsm-postperf/build/ufsm}; P=${P:-/vesuvius/ufsm/unroll/PHercParis4-u1}; O=${1:-/vesuvius/ufsm/tmp-postperf/gpu}; GPU=${GPU:-0}; RMEM=${RMEM:-7}; RCAP=${RCAP:-13G}
INK=$P/ink/ift3-24000.zarr; CTR=/vesuvius/ufsm/v2/ct; CTK=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2; LUM=9.6
FILLX="--fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 1"; S=(ws010 ws011 ws012)
[ -x "$U" ] || { echo "no $U (cmake --preset release && cmake --build build -j 8 in the branch)"; exit 1; }
mkdir -p $O/seq $O/par $O/logs || exit 1
rend() {   # rend STRIPE OUTDIR
  local s=$1 d=$2; mkdir -p $d/in_$s; ln -sfn $P/strip/strip/flat/stripes/$s $d/in_$s/$s
  CUDA_VISIBLE_DEVICES=$GPU UFSM_RENDER_GPU_MEM=4 UFSM_RENDER_MEM=$RMEM ~/bin/memcap -m $RCAP /usr/bin/time -v nice -n 10 $U render $d/in_$s $d --volume $INK $LUM \
    --ct $CTR $CTK --depth 1 --scale 1 --threads 20 $FILLX > $O/logs/$(basename $d)_$s.log 2>&1 || { echo "render $s ($d) FAILED: $O/logs/$(basename $d)_$s.log"; return 1; }
  rm -r $d/in_$s
}
t0=$(date +%s); for s in "${S[@]}"; do rend $s $O/seq || exit 1; done; t1=$(date +%s)
pids=(); for s in "${S[@]}"; do rend $s $O/par & pids+=($!); done; rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; t2=$(date +%s)
[ $rc = 0 ] || exit 1
echo "stretch renders: one after another $((t1 - t0)) s, at once $((t2 - t1)) s (the run: 84 s from 'stripimg start' to its stretch line)"
nd=0; for f in $(cd $O/seq && ls); do cmp -s $O/seq/$f $O/par/$f || { echo "DIFFERS: $f"; nd=$((nd + 1)); }; done
mkdir -p $O/fillmap; for s in "${S[@]}"; do cmp -s $O/par/${s}_fill.tif $P/strip/fillmap/${s}_fill.tif || { echo "fill map $s differs from the run's"; nd=$((nd + 1)); }; mv $O/par/${s}_fill.tif $O/fillmap/; done
st=$($U strip-mosaic $P/strip/strip/flat $O/par $O/img --stretch-only 1 --threads 3 2> $O/logs/stretch.log | awk '/^stretch/ {print $2}')
echo "stretch $st (the run: $(cat $P/strip/stretch.txt))"; [ "$st" = "$(cat $P/strip/stretch.txt)" ] || nd=$((nd + 1))
echo "renders: $nd differences"
# 2. wrap-snap with the volcomp decode on the GPU (unroll.sh SNAPGPU): the run's wraps 300..320 (their neighbours bound them) on 8 threads,
#    CPU decode vs GPU decode; every .sfc equal to the run's wraps_snap (CPU-only on the branch: 85 s, evidence 68 s of which decode 30 s)
F=/vesuvius/ufsm/tmp-postperf/p4in/fields/lat; [ -e $F/pc/zarr.json ] || F=$P/fields/lat
for mode in cpu gpu; do
  ts=$(date +%s); env $([ $mode = gpu ] && echo "UFSM_GPU_DECODE=1 UFSM_GPU_DECODE_DEV=$GPU CUDA_VISIBLE_DEVICES=$GPU" || echo "CUDA_VISIBLE_DEVICES=") ~/bin/memcap -m 12G $U wrap-snap $P/wraps $O/snap_$mode \
    --target mix --axis $P/axis.json --axis-um 2.400 --ct $CTR $CTK --ct-um $LUM --fields $F --threads 8 --first 300 --last 320 > $O/logs/snap_$mode.log 2>&1 || { echo "snap $mode FAILED"; nd=$((nd + 1)); continue; }
  d=0; for f in $O/snap_$mode/*.sfc; do cmp -s $f $P/wraps_snap/$(basename $f) || d=$((d + 1)); done
  echo "snap ($mode decode): $(( $(date +%s) - ts )) s, $(grep -o 'evidence [0-9.]* s (volume decode [0-9.]* s' $O/logs/snap_$mode.log)), $d of $(ls $O/snap_$mode | wc -l) .sfc differ from the run"; nd=$((nd + d))
done
# 3. the gpu-tagged suite test touched by the branch
( cd $(dirname $U)/.. && { cmake --preset verify && ~/bin/memcap -m 16G cmake --build build-verify -j 8 --target test_fit_join; } > $O/logs/make_test_fit_join.log 2>&1 && CUDA_VISIBLE_DEVICES=$GPU ./build-verify/test_fit_join > $O/logs/test_fit_join.log 2>&1 ) \
  && echo "test_fit_join: pass" || { echo "test_fit_join: FAIL ($O/logs/test_fit_join.log)"; nd=$((nd + 1)); }
exit $((nd > 0))

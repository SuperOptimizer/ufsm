#!/usr/bin/env bash
# postperf_gpu.sh [OUT]: the GPU-window checks of postperf-20261112 (the post-slab chain speed-ups), ~4 min on one GPU of the desk:
#   1. stripimg's stretch renders: the Paris 4 u1 run's 3 middle stripes (ws010..ws012) rendered one after another (the old schedule)
#      and at once (unroll.sh SJOBS=3), wall times of both; every output file of the two must be identical, and each fill map equal to
#      the run's strip/fillmap/<s>_fill.tif; strip-mosaic --stretch-only on them must print the run's stretch (strip/stretch.txt)
#   2. test_fit_join (tests/suite.txt, tag gpu: fit-join's recycled field blocks / sampling run there too)
# Reads the run read-only (P), writes only under OUT. Env: U (ufsm of the branch), P (the run), GPU (CUDA device, 0), RMEM (UFSM_RENDER_MEM
# GB per render, 12: three at once stay inside the 64 GB dev slice; the run used 16, a cache size: the same images).
set -u
U=${U:-$HOME/ufsm-postperf/build/ufsm}; P=${P:-/vesuvius/ufsm/unroll/PHercParis4-u1}; O=${1:-/vesuvius/ufsm/tmp-postperf/gpu}; GPU=${GPU:-0}; RMEM=${RMEM:-12}
INK=$P/ink/ift3-24000.zarr; CTR=/vesuvius/ufsm/v2/ct; CTK=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2; LUM=9.6
FILLX="--fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 1"; S=(ws010 ws011 ws012)
[ -x "$U" ] || { echo "no $U (make -j8 build/ufsm in the branch)"; exit 1; }
mkdir -p $O/seq $O/par $O/logs || exit 1
rend() {   # rend STRIPE OUTDIR
  local s=$1 d=$2; mkdir -p $d/in_$s; ln -sfn $P/strip/strip/flat/stripes/$s $d/in_$s/$s
  CUDA_VISIBLE_DEVICES=$GPU UFSM_RENDER_GPU_MEM=4 UFSM_RENDER_MEM=$RMEM ~/bin/memcap -m 20G /usr/bin/time -v nice -n 10 $U render $d/in_$s $d --volume $INK $LUM \
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
# 2. the gpu-tagged suite test touched by the branch
( cd $(dirname $U)/.. && ~/bin/memcap -m 16G make -j8 VERIFY=1 build-verify/test_fit_join > $O/logs/make_test_fit_join.log 2>&1 && CUDA_VISIBLE_DEVICES=$GPU ./build-verify/test_fit_join > $O/logs/test_fit_join.log 2>&1 ) \
  && echo "test_fit_join: pass" || { echo "test_fit_join: FAIL ($O/logs/test_fit_join.log)"; nd=$((nd + 1)); }
exit $((nd > 0))

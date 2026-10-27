#!/usr/bin/env bash
# refs.sh OUT Z0 NZ [FIELDS]: the coarse stage of the coarse-to-fine unwrap over field cells [Z0, Z0+NZ) of the union crop:
#   OUT/ref-cf  ("cf-coarse": count-field --R 0 --model-only, solve cells CFP (8) field cells: one global LS gauge)
#   OUT/ref-fit ("fitter-coarse": fit --coarse-pool FP (4), gradient terms only, lattices FLAT)
# each scored on the q-9p6 codes (OUT/<ref>.score.log). Env GPU (1), TH (12), CFP, FP, FLAT, ONLY (cf | fit), MEMGB (count-field chunk
# budget, 24: ~22 B per cell held; small chunks re-read the 128-row zarr shards: 6 GB = 16-row chunks read ~8x the data), MSTRIDE.
set -u; . $(dirname $0)/env.sh; O=$1; Z0=$2; NZ=$3; FI=${4:-$F96}; mkdir -p $O
[ -e $O/refcrop/fields.json ] || bash $MK $FI $O/refcrop $Z0 $Y0 $X0 $NZ $NY $NX > /dev/null
if [ "${ONLY:-cf}" = cf ] || [ -z "${ONLY:-}" ]; then
  [ -e $O/ref-cf/model.json ] || { NEED=50 memok; /usr/bin/time -v nice -n 10 $U count-field $O/refcrop $O/cf.unused --R 0 --pool ${CFP:-8} --phase-w 4 --iters 1200 \
      --model $O/ref-cf --model-only 1 --model-stride ${MSTRIDE:-1} --threads ${TH:-12} --mem-gb ${MEMGB:-24} > $O/ref-cf.log 2>&1; }
fi
if [ "${ONLY:-fit}" = fit ] || [ -z "${ONLY:-}" ]; then
  [ -e $O/ref-fit/model.json ] || /usr/bin/time -v $U fit $O/refcrop --coarse-pool ${FP:-4} --dir 1 --gpu ${GPU:-1} --lattice ${FLAT:-128,64,32} --out $O/ref-fit > $O/ref-fit.log 2>&1
fi
for r in ref-cf ref-fit; do [ -e $O/$r/model.json ] && [ ! -s $O/$r.score.log ] && $U fit $O/refcrop --model $O/$r --gpu ${GPU:-1} --codes $Q 9.6 --score-pairs 1 > $O/$r.score.log 2>&1; done
grep -H "ids right\|pair winding" $O/*.score.log

#!/usr/bin/env bash
# slabperf_gpu.sh [GPU]: the GPU-window checks of slabperf-20261112 (fit --crop-material, the phase gate's W_ref on the device, FITLOCK
# lanes), ~15 min on one 16 GB card (5060 Ti), all output under $OUT (default /vesuvius/ufsm/tmp-slabperf/gpu-MMDD-HHMM); reads the
# P4 / 1667 run dirs, writes nothing there.
#   1. tests (build-verify, VERIFY=1 builds of this tree): test_fit (UFSM_TEST_QUICK=1: the crop case on the GPU, host and streamed
#      fields), test_fit_join (the GPU phase gate), test_fit_lat, test_fit_w4                                              ~6 min
#   2. identity on real slabs (P4 s010 s070 s140, 1667 s030; the lane's options): BASE (origin/unrollperf-20261112) vs NEW without crop
#      vs NEW --crop-material 2 vs NEW crop + UFSM_PG_HOSTW=1 (W_ref through the host) vs NEW crop + UFSM_FIT_STREAM_VERIFY=1:
#      delta.f32 + model.json byte-compared to BASE; the stage times of each                                               ~4 min
#   3. lanes on P4 s066..s071 (one GPU): BASE 1 lane, NEW crop 1 lane, NEW crop 2 lanes + UFSM_FIT_GPU_LOCK (FITLANES=2 FITLOCK=1 in
#      unroll.sh), the same + UFSM_FIT_LOCK_LATE=1 (FITLOCK=2), wall per config, models against the 1-lane NEW ones          ~5 min
# Env: BASE, NEW (binaries; default the copies in /vesuvius/ufsm/tmp-slabperf/bin), SKIP_TESTS=1, PARTS ("1 2 3"), FTH (16: fit threads).
# Verdict lines: "IDENTICAL a b" / "DIFFER a b"; the summary at the end ($OUT/summary.txt).
set -u
G=${1:-0}; T=/vesuvius/ufsm/tmp-slabperf; O=${OUT:-$T/gpu-$(date +%m%d-%H%M)}; mkdir -p $O
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BASE=${BASE:-$T/bin/ufsm.base}; NEW=${NEW:-$T/bin/ufsm.slabperf}; FTH=${FTH:-16}; PARTS=${PARTS:-1 2 3}
P4=/vesuvius/ufsm/unroll/PHercParis4-u1; P7=/vesuvius/ufsm/unroll/PHerc1667-u1; MK=$REPO/scripts/wbench/mkcrop.sh
export CUDA_VISIBLE_DEVICES=$G   # (the fits use --gpu 0, the tests device 0)
S=$O/summary.txt; : > $S
say() { echo "$*" | tee -a $S; }
say "slabperf_gpu: GPU $G, BASE $BASE, NEW $NEW, out $O, $(date)"
# view SCROLLDIR SLAB: the slab's fit-field view (mkcrop symlinks, as unroll.sh's lane) at $O/v-<scroll>-<slab>
view() {
  local r=$1 s=$2 v=$O/v-$(basename $1)-$2 c0 c1 a b ny nx
  [ -e $v/fields.json ] || { read -r c0 c1 a b < $r/slabs/$s/cells; ny=$(jq -r '.shape[1]' $r/fields/lat/fields.json); nx=$(jq -r '.shape[2]' $r/fields/lat/fields.json)
                             bash $MK $r/fields/lat $v $a 0 0 $((b - a)) $ny $nx > /dev/null || return 1; }
  echo $v
}
# fit NAME BIN SCROLLDIR VIEW [extra..]: one slab fit with the lane's options (fit.log -> NAME.log, model -> NAME/)
fit() {
  local n=$1 b=$2 r=$3 v=$4; shift 4
  env UFSM_FIT_THREADS=$FTH /usr/bin/time -f "wall %e s, max RSS %M KB" nice -n 5 $b fit $v --gpu 0 --out $O/$n --dir $(jq -r .dir $r/ref-cf/model.json) --ref-model $r/ref-cf \
    --ref-snap 1 --uw-repair 0 --steps 300,300,300,300,200,200 --batch 1000000 "$@" > $O/$n.log 2>&1
  local rc=$?; echo "  $n: rc $rc, $(tail -1 $O/$n.log)" >> $S
  grep -E "managed|material crop|loaded in|phase gate|voxel unwrapping [0-9]|stage [0-5]: [0-9]|stream verify|illegal|error" $O/$n.log | sed 's/^/      /' >> $S
  return $rc
}
same() { if cmp -s $O/$1/delta.f32 $O/$2/delta.f32 && cmp -s $O/$1/model.json $O/$2/model.json; then say "IDENTICAL $1 $2"; else say "DIFFER $1 $2"; fi; }

for part in $PARTS; do case $part in
1) [ "${SKIP_TESTS:-0}" = 1 ] && continue
   say "== 1. tests"
   for t in test_fit_lat test_fit_join test_fit_w4; do (cd $REPO && ./build-verify/$t > $O/$t.log 2>&1); say "$t rc $? $(tail -1 $O/$t.log)"; done
   (cd $REPO && UFSM_TEST_QUICK=1 ./build-verify/test_fit > $O/test_fit.log 2>&1); say "test_fit rc $? $(tail -1 $O/test_fit.log)"
   grep -E "^crop|FAIL" $O/test_fit.log | sed 's/^/  /' | tee -a $S ;;
2) say "== 2. identity on real slabs"
   for rs in "$P4 s070" "$P4 s010" "$P4 s140" "$P7 s030"; do
     set -- $rs; r=$1 s=$2; k=$(basename $r | cut -c1-9)-$s; v=$(view $r $s) || { say "no view $r $s"; continue; }
     fit $k.base $BASE $r $v; fit $k.nocrop $NEW $r $v --crop-material 0; fit $k.crop $NEW $r $v --crop-material 2
     UFSM_PG_HOSTW=1 fit $k.hostw $NEW $r $v --crop-material 2; UFSM_FIT_STREAM_VERIFY=1 fit $k.sverify $NEW $r $v --crop-material 2
     for x in nocrop crop hostw sverify; do same $k.base $k.$x; done
     [ -e $r/slabs/$s/fit/delta.f32 ] && { cmp -s $r/slabs/$s/fit/delta.f32 $O/$k.base/delta.f32 && say "  (the run's own $s fit equals BASE)" || say "  (the run's own $s fit differs from BASE: another binary)"; }
   done ;;
3) say "== 3. lanes on P4 s066..s071"
   L="s066 s067 s068 s069 s070 s071"; for s in $L; do view $P4 $s > /dev/null; done
   t0=$(date +%s.%N); for s in $L; do fit l1b-$s $BASE $P4 $O/v-PHercParis4-u1-$s; done; t1=$(date +%s.%N)
   for s in $L; do fit l1c-$s $NEW $P4 $O/v-PHercParis4-u1-$s --crop-material 2; done; t2=$(date +%s.%N)
   lane2() { local s; for s in "$@"; do UFSM_FIT_GPU_LOCK=$O/gpu.lock fit l2c-$s $NEW $P4 $O/v-PHercParis4-u1-$s --crop-material 2; done; }
   lane2 s066 s068 s070 & p1=$!; lane2 s067 s069 s071 & p2=$!; wait $p1 $p2; t3=$(date +%s.%N)
   lane3() { local s; for s in "$@"; do UFSM_FIT_GPU_LOCK=$O/gpu.lock UFSM_FIT_LOCK_LATE=1 fit l3c-$s $NEW $P4 $O/v-PHercParis4-u1-$s --crop-material 2; done; }
   lane3 s066 s068 s070 & p1=$!; lane3 s067 s069 s071 & p2=$!; wait $p1 $p2; t4=$(date +%s.%N)
   say "$(awk -v a=$t0 -v b=$t1 -v c=$t2 -v d=$t3 -v e=$t4 'BEGIN { printf "6 slabs: BASE 1 lane %.1f s, NEW crop 1 lane %.1f s, 2 lanes + GPU lock (FITLOCK=1) %.1f s, + late lock (FITLOCK=2) %.1f s", b - a, c - b, d - c, e - d }')"
   for s in $L; do same l1b-$s l1c-$s; same l1c-$s l2c-$s; same l1c-$s l3c-$s; done ;;
esac; done
say "slabperf_gpu: done $(date); $(grep -c '^DIFFER' $S) DIFFER, $(grep -c '^IDENTICAL' $S) IDENTICAL, test failures: $(grep -cE '^test_.* rc [1-9]' $S)"

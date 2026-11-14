#!/usr/bin/env bash
# lab_early under --split z (2026-11-14): the next batch's label warp after its input, under the forward, in chunks through hs_lg
# (nn_lw_sink). Both halves on one GPU (--gpus 0,0, the split's emulation), the wind fixture with ink and fibre targets (--ink-head 1 --fib-head 1) and --rv 2 (the papyrus
# plane from ct8 + known bits): seven label planes, the latent grid with the guided head, --gpu-aug 3 (w4 runs), --det 1:
#  (1) UFSM_LAT_STREAM=check: every head slab's streamed planes compared byte for byte with the device-built ones (the trainer aborts on
#      a difference), the "label warp: after the input" line printed;
#  (2) the logged losses equal those of UFSM_LAB_EARLY=0 (the warp at the step boundary into the input buffer), line for line;
#  (3) (1) and (2) with the planes in pinned host memory (UFSM_LAT_LDEV=0), not packed on the device.
# Usage: test_lab_early_split.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 128 --B 1 --gpus 0,0 --split z --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2
   --gpu-aug 3 --warmup 0 --seed 5 --widths 16,32,64,64 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --sched wsd --cooldown 0 --opt muon
   --muon-lr 0.002 --lr 0.001 --packed-weight 8 --ink-head 1 --fib-head 1 --steps 6)
fail() { echo "FAIL: $*"; exit 1; }
run() { local tag=$1; shift; env "$@" "$U" train fx/sources-ink.json --out $tag "${C[@]}" > $tag.log 2>&1 || fail "$tag ($D/$tag.log)"; }
losses() { grep '^step' $1.log | sed 's/ [0-9.]* samp\/s.*//'; }
run early UFSM_LAT_STREAM=check
grep -q "^label warp: after the input, under the forward (--split z" early.log || fail "lab_early not taken under --split z ($D/early.log)"
[ "$(grep -c '^step' early.log)" = 6 ] || fail "not 6 steps ($D/early.log)"
run late UFSM_LAB_EARLY=0
grep -q "^label warp: at the step boundary" late.log || fail "UFSM_LAB_EARLY=0 did not switch it off ($D/late.log)"
[ "$(losses early)" = "$(losses late)" ] || fail "losses differ with the early label warp ($D/early.log, $D/late.log)"
run early2
[ "$(losses early)" = "$(losses early2)" ] || fail "losses differ with UFSM_LAT_STREAM=check ($D/early.log, $D/early2.log)"
# (3) the same with every plane in pinned host memory (UFSM_LAT_LDEV=0: the desk's case, each chunk D2H into the host planes)
run hearly UFSM_LAT_LDEV=0 UFSM_LAT_STREAM=check
run hlate UFSM_LAT_LDEV=0 UFSM_LAB_EARLY=0
[ "$(losses hearly)" = "$(losses hlate)" ] || fail "host planes: losses differ with the early label warp ($D/hearly.log, $D/hlate.log)"
echo "test_lab_early_split: ok (check mode equal on every slab, losses equal to the step-boundary warp)"

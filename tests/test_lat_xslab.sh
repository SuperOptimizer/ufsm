#!/usr/bin/env bash
# UFSM_LAT_XSLAB=1 (2026-11-27, vramperf): the latent net's full-resolution CT channel in the level-0 gradient buffer from its upload to the
# forward's input conversion, the head slabs' planes written again from the kept scanned bytes (nn_aug_rewrite). The wind fixture, one GPU,
# --lat 4 with the guided head, the fine stem, rv + ink + fibre heads, --gpu-aug 3 with scanner and geometry augmentation, level 0 256 wide
# (its MX-fp4 gradient holds 2 B per voxel), lean 2 forced by a tiny planned free memory, 8-plane head slabs, --det 1, validation every 3 steps:
#  (1) the on line, no DEFAULTED for it; (2) the logged losses (training and validation) equal those of the run without it, line by line;
#  (3) the step-start hashes of the input channel and the label planes (UFSM_DET_TRACE) equal (the parameters' hash differs run to run here,
#      off against off too); UFSM_LAT_XSLAB=check: every slab's rewritten planes equal the uploaded channel's bytes; (4) the plan's lat batch term shrinks by ~1 B per voxel.
# Usage: test_lat_xslab.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 128 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 3 --val-train-ok 1 --log-every 1
   --ckpt-every 1000 --fp4 2 --gpu-aug 3 --warmup 0 --seed 5 --widths 256,64,64 --det 1 --lat 4 --lat-guide 3 --fstem 16 --rv 2 --rv-iso 100
   --ink-head 1 --fib-head 1 --sched wsd --cooldown 0 --opt muon --muon-lr 0.002 --lr 0.001 --packed-weight 8 --steps 7
   --aug-blur-p 0.7 --aug-noise-p 0.5 --aug-tone-p 0.5 --aug-ring-p 0.3 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.5 --elastic-p 0.5)
fail() { echo "FAIL: $*"; exit 1; }
run() { local tag=$1; shift; env UFSM_MEM_FREE_GB=0.05 UFSM_HEAD_SLAB_Z=8 UFSM_DET_TRACE=1 UFSM_MEM_REPORT=2 "$@" "$U" train fx/sources-ink.json --out $tag "${C[@]}" > $tag.log 2>&1 || fail "$tag ($D/$tag.log)"; }
losses() { sed -E 's/^t\+[0-9.]+s //' $1.log | grep -E '^(step|val)' | sed -E 's/ [0-9.]+ samp\/s.*//; s/ \([0-9.]+ s\)//'; }
run off
run on UFSM_LAT_XSLAB=1
run chk UFSM_LAT_XSLAB=check
[ "$(grep -c 'lat xslab check: .* 0 of [0-9]* slab bytes differ' chk.log)" -gt 0 ] && ! grep 'lat xslab check' chk.log | grep -vq ' 0 of ' || { grep 'lat xslab check' chk.log | tail -2; fail "rewritten slab planes differ from the uploaded channel ($D/chk.log)"; }
grep -q 'latent grid: UFSM_LAT_XSLAB: the CT channel in the level-0 gradient buffer, the kept bytes' on.log || fail "UFSM_LAT_XSLAB not taken ($D/on.log: $(grep XSLAB on.log | tail -1))"
! grep -q 'DEFAULTED: .*XSLAB' on.log || fail "UFSM_LAT_XSLAB defaulted ($D/on.log)"
grep -q 'lean 2' on.log || fail "not a lean 2 mode ($D/on.log)"
[ "$(losses on | grep -c '^step')" = 7 ] || fail "not 7 steps ($D/on.log)"
[ "$(grep -c '^val' on.log)" -ge 2 ] || fail "fewer than 2 validation lines ($D/on.log)"
[ "$(losses off)" = "$(losses on)" ] || { diff <(losses off) <(losses on) | head -6; fail "losses differ ($D/off.log, $D/on.log)"; }
xh() { grep '^dettrace' $1.log | grep -o ' x [0-9a-f]* labels [0-9a-f ]*'; }   # (the parameters' hash differs run to run under --det here, off against off too)
[ "$(xh off)" = "$(xh on)" ] || { diff <(xh off) <(xh on) | head -4; fail "step-start input / label hashes differ"; }
grep -q 'lat batch [0-9.]* (input kept bytes + slab planes (UFSM_LAT_XSLAB)' on.log || fail "the plan does not count the kept bytes ($D/on.log)"
grep -q 'lat batch [0-9.]* (input own buffer' off.log || fail "the plan without it does not count the input's own buffer ($D/off.log)"
echo "test_lat_xslab: ok ($(losses on | grep -c '^step') steps + $(grep -c '^val' on.log) validation lines equal, input / label hashes equal, every rewritten slab byte equal)"

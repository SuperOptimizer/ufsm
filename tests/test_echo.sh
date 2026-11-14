#!/usr/bin/env bash
# train --echo-max (adaptive data echoing, 2026-11-14) on the wind fixture, one GPU, the latent grid with the rv pair and the guided head:
#  (1) one sampler worker taking 250 ms more per window (UFSM_SAMPLER_DELAY_MS: starved whatever the GPU's speed; without it a GPU shared
#      by the parallel suite kept up and nothing was echoed): the startup line, " echo N%" on the step lines with N > 0 overall,
#      every step logged, finite losses;
#  (2) --det 1: echo is switched off with its message, and the losses equal those of the same --det run without --echo-max, line by line.
# Usage: test_echo.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --sampler-slots 4 --val-batches 0 --log-every 2 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --sched wsd --cooldown 0 --opt muon
   --muon-lr 0.002 --lr 0.001 --packed-weight 8)
fail() { echo "FAIL: $*"; exit 1; }
run() { local tag=$1; shift; env ${ENV:-} "$U" train fx/sources.json --out $tag "${C[@]}" "$@" > $tag.log 2>&1 || fail "$tag ($D/$tag.log)"; }
losses() { grep '^step' $1.log | sed 's/ [0-9.]* samp\/s.*//'; }
ENV=UFSM_SAMPLER_DELAY_MS=250 run starve --workers 1 --steps 40 --echo-max 3
grep -q "^sampler: adaptive echo" starve.log || fail "no adaptive echo line ($D/starve.log)"
[ "$(grep -c '^step' starve.log)" = 20 ] || fail "not 20 step lines ($D/starve.log)"
! grep '^step' starve.log | grep -qiE 'nan|inf ' || fail "non-finite losses ($D/starve.log)"
e=$(grep '^step' starve.log | grep -o ' echo [0-9]*%' | tr -dc '0-9\n' | awk '{s += $1} END {print s + 0}')
[ "$e" -gt 0 ] || fail "no echoes with one worker ($D/starve.log)"
echo "  starved: $(grep '^step' starve.log | grep -c ' echo ') lines with an echo share (sum $e %), losses finite"
run det0 --workers 2 --steps 8 --det 1
run det1 --workers 2 --steps 8 --det 1 --echo-max 3
grep -q "echo-max off under --det 1" det1.log || fail "--det 1 did not switch echo off ($D/det1.log)"
[ "$(losses det0)" = "$(losses det1)" ] || fail "--det 1 losses differ with --echo-max ($D/det0.log, $D/det1.log)"
echo "  --det 1: echo off, losses equal"
echo "test_echo: ok"

#!/usr/bin/env bash
# test_serialize (2026-11-14, ops audit #9): a stream-ordering race shows as a difference between the default run and a serialised
# one. UFSM_SERIALIZE=1 sets CUDA_LAUNCH_BLOCKING=1 at nn_init: every kernel launch and copy completes before the host continues, so
# the GPU work runs in program order on every stream (the semantics the code means). Under --det the default run must print the same
# step lines. 1bb0db8 (ink MIL rays overwritten under the backward) was this class, found in production.
# Covered: the universal recipe (latent grid, rv, rungs, film, fstem, uw, grad-cos, gpu-aug 3, replay) data parallel on two (logical)
# devices; the ink MIL path has the same check in test_ink_mil_train.
# Usage: test_serialize.sh UFSM MAKE_WIND_FIXTURE DIR   (env UFSM_TEST_GPUS: the DP device list, default 0,1)
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3; G2=${UFSM_TEST_GPUS:-0,1}
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(fx/sources-l1.json --task unwrap2 --P 64 --B 1 --workers 2 --sampler-slots 4 --val-batches 2 --val-train-ok 1 --val-every 4 --log-every 1 --ckpt-every 1000
   --fp4 2 --gpu-aug 3 --replay 3 --warmup 0 --seed 5 --widths 16,32,64 --rv 2 --rv-iso 100 --lat 4 --lat-guide 3 --um-rungs 19.2,38.4
   --film 2 --fstem 16 --uw 1 --grad-cos 3 --det 1 --steps ${SERN:-8})
fails=0
run() {   # NAME DEVICES [extra...]: default and serialised, step / val lines compared (timing stripped)
  local n=$1 g=$2; shift 2
  "$U" train "${C[@]}" --gpus "$g" --out "$n.a" "$@" > "$n.a.log" 2>&1 || { echo "FAIL: $n default run ($D/$n.a.log)"; tail -4 "$n.a.log"; fails=$((fails + 1)); return; }
  UFSM_SERIALIZE=1 CUDA_LAUNCH_BLOCKING=1 "$U" train "${C[@]}" --gpus "$g" --out "$n.s" "$@" > "$n.s.log" 2>&1 || { echo "FAIL: $n serialised run ($D/$n.s.log)"; tail -4 "$n.s.log"; fails=$((fails + 1)); return; }
  grep -q 'UFSM_SERIALIZE' "$n.s.log" || { echo "FAIL: $n: the serialised run did not say so"; fails=$((fails + 1)); return; }
  for x in a s; do grep -E '^step|^val' "$n.$x.log" | sed -E 's/ [0-9.]+ samp\/s.*//; s/ (wait|gn) [0-9.%]+//g' > "$n.$x.txt"; done
  [ "$(grep -c '^step' "$n.a.txt")" -ge 4 ] || { echo "FAIL: $n: $(grep -c '^step' "$n.a.txt") step lines"; fails=$((fails + 1)); return; }
  if cmp -s "$n.a.txt" "$n.s.txt"; then echo "ok: $n: $(grep -c '^step' "$n.a.txt") step lines identical, default vs serialised"
  else echo "FAIL: $n: default and serialised runs differ (a stream-ordering race?):"; diff "$n.a.txt" "$n.s.txt" | head -6; fails=$((fails + 1)); fi
}
run dp "$G2"
[ $fails = 0 ] && echo "test_serialize: ok" || { echo "test_serialize: $fails failure(s)"; exit 1; }

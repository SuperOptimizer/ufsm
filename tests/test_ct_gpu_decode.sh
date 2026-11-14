#!/bin/bash
# train --ct-gpu-decode (DESIGN 2026-11-13): the sampler's volcomp CT decode on the GPU (src/nn_gpudec.cu through z3_read).
#  1. ufsm read of the fixture CT (every level, whole array) with UFSM_GPU_DECODE=1 = the CPU decode, byte for byte, and the GPU decoded
#     substreams (UFSM_GPU_DECODE_STATS > 0 subs), so the comparison is not CPU vs CPU.
#  2. train --ct-gpu-decode 0 runs: the startup line, the decoder's statistics at exit with substreams decoded on the GPU, and under
#     --det 1 the same batches as the CPU-decode run: every batch plane's hash (UFSM_BATCH_HASH) and the device input of every step
#     (UFSM_DET_TRACE "x") equal. Losses only within 2%: --det fixes the windows, not the GPU's float atomics (GN / bias sums,
#     split-K weight gradients), so two runs of the same binary differ in the last bits of the gradients and the step-4 loss then
#     takes one of two values (2.3195 / 2.3271 at random, also CPU vs CPU; 2026-11-13).
# Usage: test_ct_gpu_decode.sh <ufsm binary> <make_wind_fixture binary>
set -u
U=$1 F=$2
D=$(mktemp -d "${TMPDIR:-/tmp}/test_ctgd_XXXXXX")
fail() { echo "FAIL: $*"; echo "kept $D"; exit 1; }
"$F" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1 || fail "fixture"
for k in $(ls "$D/fx/ct" | grep -v json); do
  read -r nz ny nx < <("$U" info "$D/fx" "ct/$k" | awk '/shape/ {print $2, $4, $6}')
  UFSM_GPU_DECODE=0 "$U" read "$D/fx" "ct/$k" 0 0 0 "$nz" "$ny" "$nx" "$D/cpu.raw" > /dev/null 2>&1 || fail "cpu read ct/$k"
  UFSM_GPU_DECODE=1 UFSM_GPU_DECODE_STATS=1 "$U" read "$D/fx" "ct/$k" 0 0 0 "$nz" "$ny" "$nx" "$D/gpu.raw" > "$D/gpu.log" 2>&1 || fail "gpu read ct/$k"
  cmp -s "$D/cpu.raw" "$D/gpu.raw" || fail "ct/$k: the GPU decode differs from the CPU decode"
  subs=$(grep -oE '[0-9]+ substreams' "$D/gpu.log" | head -1 | awk '{print $1}')
  echo "ct/$k ($nz x $ny x $nx): GPU = CPU byte for byte, ${subs:-?} substreams on the GPU"
done
grep -qE '[1-9][0-9]* substreams' "$D/gpu.log" || { cat "$D/gpu.log"; fail "no substream decoded on the GPU (gpudec statistics)"; }
T="--det 1 --task unwrap2 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --P 64 --widths 32,64,128 --steps 4"
# shellcheck disable=SC2086
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "$U" train "$D/fx/sources.json" --out "$D/cpu" $T > "$D/cpu.tlog" 2>&1 || fail "train, CPU decode ($D/cpu.tlog)"
# shellcheck disable=SC2086
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "$U" train "$D/fx/sources.json" --out "$D/gpu" $T --ct-gpu-decode 0 > "$D/gpu.tlog" 2>&1 || fail "train --ct-gpu-decode 0 ($D/gpu.tlog)"
grep -q 'CT decode (volcomp substreams) on GPU 0' "$D/gpu.tlog" || fail "no --ct-gpu-decode startup line"
grep -qE '[1-9][0-9]* substreams' "$D/gpu.tlog" || fail "train: no substream decoded on the GPU ($D/gpu.tlog)"
ha=$(grep -E '^batchhash' "$D/cpu.tlog"); hb=$(grep -E '^batchhash' "$D/gpu.tlog")
[ -n "$ha" ] || fail "no batchhash lines"
[ "$ha" = "$hb" ] || { diff <(echo "$ha") <(echo "$hb") | head -6; fail "the batches differ between the CPU and the GPU CT decode"; }
xa=$(grep -E '^dettrace' "$D/cpu.tlog" | awk '{print $3, $5}'); xb=$(grep -E '^dettrace' "$D/gpu.tlog" | awk '{print $3, $5}')
[ -n "$xa" ] || fail "no dettrace lines"
[ "$xa" = "$xb" ] || { echo "cpu: $xa"; echo "gpu: $xb"; fail "the device input differs between the CPU and the GPU CT decode"; }
a=$(grep -E '^step' "$D/cpu.tlog" | awk '{print $6}'); b=$(grep -E '^step' "$D/gpu.tlog" | awk '{print $6}')
[ -n "$a" ] && [ "$(echo "$a" | wc -l)" = "$(echo "$b" | wc -l)" ] || fail "no / unequal step lines"
paste <(echo "$a") <(echo "$b") | awk '{d = $1 - $2; if (d < 0) d = -d; if (d > 0.02 * $1) bad = 1} END {exit bad}' || { echo "cpu: $a"; echo "gpu: $b"; fail "train losses differ by more than 2%"; }
echo "train: --ct-gpu-decode 0 and the CPU decode: the same $(echo "$ha" | wc -l) batches and device inputs over $(echo "$xa" | wc -l) steps, losses within 2%"
rm -rf "$D"; echo "test_ct_gpu_decode: ok"

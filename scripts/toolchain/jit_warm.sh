#!/usr/bin/env bash
# jit_warm.sh BIN [N]: fill mcudahw's kernel cache (~/.cache/mcudahw, or MCUDA_CACHE) for an mcc-built BIN, so no kernel is JIT-compiled
# by the driver inside a training step or a predict. mcudahw hands each kernel's PTX to the driver's ptxas at its first launch; the cache
# is keyed by the PTX, the GPU's name and the driver version, so a new machine, a new GPU model, a new driver or a rebuilt kernel starts
# cold (~4600 kernels, ~350 MB of PTX in a release build; lp_f4fwd_i1 alone ~70 s of ptxas). MCUDA_PRECOMPILE=N compiles every kernel
# the cache lacks, N at a time, then exits: a warm cache costs only the scan. Needs a GPU (a context) but no free memory to speak of.
#   N   threads (default: the cgroup's CPUs, `BIN sysinfo --ncpu`; capped at 64 by mcudahw)
# A no-op (exit 0) for an nvcc / clang build (no mcuda_modules section). Env: CUDA_VISIBLE_DEVICES (the first visible GPU is compiled
# for: run once per GPU model), MCUDA_PRECOMPILE_ONLY (a kernel-name substring), JITLOG (log file, default stderr).
# Called by scripts/pod/setup.sh after the build and by the `jit-warm` CMake target of the mcc toolchain.
set -u
B=${1:?BIN}; [ -x "$B" ] || { echo "jit_warm: no $B" >&2; exit 2; }
readelf -SW "$B" 2>/dev/null | grep -q mcuda_modules || { echo "jit_warm: $B is not an mcc build (no mcuda_modules): nothing to do"; exit 0; }
N=${2:-$("$B" sysinfo --ncpu 2>/dev/null || nproc)}; [[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -ge 1 ] || N=4
C=${MCUDA_CACHE:-$HOME/.cache/mcudahw}; n0=$(ls "$C" 2>/dev/null | wc -l)
L=${JITLOG:-/dev/stderr}; t0=$(date +%s)
MCUDA_PRECOMPILE=$N "$B" sysinfo > /dev/null 2>> "$L"; rc=$?
n1=$(ls "$C" 2>/dev/null | wc -l)
echo "jit_warm: $(basename "$B"): rc $rc, $((n1 - n0)) kernels compiled in $(( $(date +%s) - t0 )) s at $N threads; cache $C: $n1 cubins, $(du -sm "$C" 2>/dev/null | cut -f1) MB"
exit $rc

#!/usr/bin/env bash
# test_wbench_gpulock.sh: wbench.sh's desk locks name the PHYSICAL card (lead 11-17): GPU indexes CUDA_VISIBLE_DEVICES when it is set,
# UUID entries mapped by nvidia-smi (a fake one here: no GPU needed). Also the shared-lock helper: an exclusive holder waits / gets in.
set -uo pipefail
W=scripts/wbench/wbench.sh; T=${TMPDIR:-/tmp}/wbgl.$$; mkdir -p "$T"; trap 'rm -rf -- "$T"' EXIT
eval "$(sed -n '/^gpu_phys() {/,/^}/p' $W)"; eval "$(sed -n '/^DLFD=()/,/^dulk()/p' $W)"
cat > "$T/nvsmi" <<'X'
#!/bin/sh
echo "0, GPU-249c84e8-79ed-a6dc-51bc-ea5a73ae1002"; echo "1, GPU-94355c13-dfd2-ab84-584c-514bf14fb949"
X
chmod +x "$T/nvsmi"; export NVSMI=$T/nvsmi; bad=0
ck() { local want=$1 got; shift; got=$(env "$@" bash -c "$(declare -f gpu_phys); gpu_phys \$GPU" 2> /dev/null); rc=$?
  [ "$got" = "$want" ] && [ $rc -eq 0 ] || { echo "FAIL: $* -> '$got' rc $rc (want $want)"; bad=1; }; }
ck 1 -u CUDA_VISIBLE_DEVICES GPU=1
ck 1 CUDA_VISIBLE_DEVICES=1 GPU=0                 # airiso's setup: logical 0 = card 1
ck 0 CUDA_VISIBLE_DEVICES=1,0 GPU=1
ck 1 CUDA_VISIBLE_DEVICES=GPU-94355c13-dfd2-ab84-584c-514bf14fb949 GPU=0
ck 1 CUDA_VISIBLE_DEVICES=GPU-94355c13 GPU=0        # a UUID prefix, as CUDA accepts
ck 0 "CUDA_VISIBLE_DEVICES=GPU-249c84e8,1" GPU=0
env CUDA_VISIBLE_DEVICES=1 GPU=1 bash -c "$(declare -f gpu_phys); gpu_phys 1" > /dev/null 2>&1 && { echo "FAIL: index outside CVD accepted"; bad=1; }
env CUDA_VISIBLE_DEVICES=GPU-deadbeef bash -c "$(declare -f gpu_phys); gpu_phys 0" > /dev/null 2>&1 && { echo "FAIL: unknown UUID accepted"; bad=1; }
# the lock helper: held shared -> an exclusive try fails; released -> it succeeds
: > "$T/a.lock"; DLOCKS=$T/a.lock; dlk
flock -n -x "$T/a.lock" true && { echo "FAIL: exclusive got in while held"; bad=1; }
dulk; flock -n -x "$T/a.lock" true || { echo "FAIL: lock still held after dulk"; bad=1; }
[ $bad = 0 ] && echo "test_wbench_gpulock: ok"; exit $bad

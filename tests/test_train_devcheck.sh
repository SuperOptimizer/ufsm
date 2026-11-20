#!/usr/bin/env bash
# train's start-up check of the device flags (2026-11-20, FR1: --ct-gpu-decode 1 under CUDA_VISIBLE_DEVICES=0 quietly decoded on the CPU):
# with no visible CUDA device (CUDA_VISIBLE_DEVICES empty, so this runs on CPU-only machines too) an explicit --ct-gpu-decode / --ink-gpu /
# --fibre-gpu / --onfly-gpu (either of A,B) index stops the run with rc 2 and names the flag, before any source is read; -1 passes the check;
# under UFSM_SAMPLER_ONLY the same flag is a DEFAULTED line instead, and UFSM_STRICT_DEFAULTS=1 makes that fatal (exit 3).
# Usage: test_train_devcheck.sh UFSM DIR
set -u
U=$(readlink -f "$1"); D=$2
rm -rf "$D"; mkdir -p "$D"; cd "$D" || exit 1
fail() { echo "FAIL: $*"; exit 1; }
t() { local log=$1; shift; CUDA_VISIBLE_DEVICES= "$U" train missing.json --out "$D/o" "$@" > "$log" 2>&1; echo $?; }
for f in --ct-gpu-decode --ink-gpu --fibre-gpu; do
  l=${f#--}.log; rc=$(t $l $f 0); [ "$rc" = 2 ] || fail "$f 0 with no device: rc $rc, want 2 ($D/$l)"
  grep -q -- "^train: $f 0: no such CUDA device (0 visible" $l || fail "$f 0: no such-device line ($D/$l)"
done
rc=$(t pair.log --onfly-gpu -1,3); [ "$rc" = 2 ] && grep -q "^train: --onfly-gpu 3: no such CUDA device" pair.log || fail "--onfly-gpu -1,3: rc $rc ($D/pair.log)"
rc=$(t cpu.log --ct-gpu-decode -1 --ink-gpu -1); ! grep -q 'no such CUDA device' cpu.log || fail "-1 (the CPU on purpose) refused ($D/cpu.log)"
rc=$(UFSM_SAMPLER_ONLY=1 t so.log --ink-gpu 0); grep -q '^DEFAULTED: --ink-gpu 0 = the CPU (no such CUDA device' so.log || fail "UFSM_SAMPLER_ONLY: no DEFAULTED line ($D/so.log)"
rc=$(UFSM_SAMPLER_ONLY=1 UFSM_STRICT_DEFAULTS=1 t sos.log --ink-gpu 0); [ "$rc" = 3 ] || fail "UFSM_STRICT_DEFAULTS=1: rc $rc, want 3 ($D/sos.log)"
echo "test_train_devcheck: ok"

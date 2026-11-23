#!/usr/bin/env bash
# `ufsm sample` with no visible CUDA device (hfink 2026-11-23): the batch slots fall back to plain malloc (DEFAULTED) instead of a failed
# cudaMallocHost returning null and the first window's symmetry copy writing through it (segfault in sym_u8). A winding and an ink fixture.
# Usage: test_sample_nogpu.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 1 1 > fx.log 2>&1
for s in sources sources-ink; do
  mkdir -p o-$s
  CUDA_VISIBLE_DEVICES= "$U" sample fx/$s.json --P 64 --B 2 --n 2 --workers 1 --out o-$s > $s.log 2>&1 || { echo "FAIL: ufsm sample $s rc $? ($D/$s.log)"; tail -3 $s.log; exit 1; }
  n=$(ls o-$s | grep -c '\.pgm$' || true)
  [ "$n" -eq 4 ] || { echo "FAIL: $s: $n patches (want 4)"; exit 1; }
  grep -q '^DEFAULTED: sampler batch slots = plain malloc' $s.log || { echo "FAIL: $s: no DEFAULTED line for the unpinned slots"; exit 1; }
done
echo "ok: ufsm sample without a GPU (winding + ink fixtures, 4 patches each, DEFAULTED printed)"

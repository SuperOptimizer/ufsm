#!/usr/bin/env bash
# idle workers release their scratch (2026-11-13, hostmem: sample.c worker_idle_release): a worker that finds no free slot madvises its
# window scratch away before it sleeps. Every draw writes its scratch before reading it, so the batches must not change: sampler-only
# --det 1 with more workers than slots (3 workers, 1 slot: two always idle; P 128: the scratch blocks above wbig_alloc's 4 MB, page-aligned, so madvised), UFSM_IDLE_RELEASE=0 vs the default, the batchhash lines equal.
# CPU only. Usage: test_idle_release.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 128 --B 1 --workers 3 --sampler-slots 1 --seed 7 --det 1 --wind4-tile 0 --widths 16,32,64 --gpu-aug 0)
for k in keep rel; do
  e=(); [ $k = keep ] && e=(UFSM_IDLE_RELEASE=0)
  env CUDA_VISIBLE_DEVICES= UFSM_SAMPLER_ONLY=12 UFSM_BATCH_HASH=1 "${e[@]}" "$U" train fx/sources.json --out $k "${C[@]}" > $k.log 2>&1 || { echo "FAIL: run $k ($D/$k.log)"; tail -3 $k.log; exit 1; }
done
grep -q 'released its scratch' rel.log || { echo "FAIL: no worker released its scratch ($D/rel.log)"; exit 1; }
! grep -q 'released its scratch' keep.log || { echo "FAIL: UFSM_IDLE_RELEASE=0 released"; exit 1; }
a=$(grep '^batchhash' keep.log | sed -E 's/ slot [0-9]+//'); b=$(grep '^batchhash' rel.log | sed -E 's/ slot [0-9]+//')
[ -n "$a" ] && [ "$a" = "$b" ] || { echo "FAIL: the batches differ"; diff <(echo "$a") <(echo "$b") | head -4; exit 1; }
echo "test_idle_release: ok ($(echo "$a" | wc -l) batchhash lines identical, 3 workers on 1 slot)"

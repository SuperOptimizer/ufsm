#!/usr/bin/env bash
# the dense label store (denselab, DESIGN 2026-11-19): the wind fixture's on-the-fly phase (levels 1 and 2: the codes and their 2x
# pooling) rendered by ufsm denselab-render; --verify: random boxes of every rendered array, served by the store (no local cache), equal
# a fresh compute byte for byte; the sampler's batches (UFSM_BATCH_HASH, sampler only, CPU) equal with the store and on the fly; a
# store with its shards marked unrendered falls back to the compute (same batches). CPU only.
# Usage: test_denselab.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
export CUDA_VISIBLE_DEVICES=
"$MF" fx 256 2 > fx.log 2>&1 || { echo "FAIL: fixture ($D/fx.log)"; exit 1; }
F=$(readlink -f fx)
sed -E "s#\"phase\": *\\{[^}]*\\}#\"phase\": {\"onfly\": {\"root\": \"$F/band\", \"key\": \"1\", \"rmax\": 12}, \"level\": 1}#; s#\"weight\": *1,#\"weight\": 1, \"min_level\": 1, \"max_level\": 2,#" fx/sources-l1.json > onfly.json
grep -q '"onfly"' onfly.json && grep -q '"max_level": 2' onfly.json || { echo "FAIL: could not make the onfly sources json"; exit 1; }
"$U" denselab-render onfly.json --out "$D/store" --threads 4 --shard 128 > r.log 2>&1 || { echo "FAIL: render ($D/r.log)"; exit 1; }
n=$(grep -c ': done, ' r.log || true); [ "$n" -ge 2 ] || { echo "FAIL: expected >= 2 rendered arrays (phase L1, L2), got $n ($D/r.log)"; exit 1; }
"$U" denselab-render onfly.json --out "$D/store" --threads 4 --verify 6 --P 64 > v.log 2>&1 || { echo "FAIL: verify ($D/v.log)"; grep verify v.log; exit 1; }
[ "$(grep -c 'byte-identical' v.log)" -ge 2 ] || { echo "FAIL: not every array verified ($D/v.log)"; exit 1; }
"$U" denselab-render onfly.json --out "$D/store" --threads 4 > r2.log 2>&1 || { echo "FAIL: re-render ($D/r2.log)"; exit 1; }
grep -q '0 to render' r2.log || { echo "FAIL: a re-render rendered again (resume, $D/r2.log)"; exit 1; }
C=(--task unwrap2 --levels 1,2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000
   --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 4 --onfly-gpu -1)
run() { UFSM_SAMPLER_ONLY=6 UFSM_BATCH_HASH=1 UFSM_DENSELAB=$2 UFSM_DERIVED_CACHE=off "$U" train onfly.json --out "tr$1" "${C[@]}" > "tr$1.log" 2>&1 || { echo "FAIL: sampler run $1 ($D/tr$1.log)"; exit 1; }
        grep 'batchhash' "tr$1.log" | sort > "h$1.txt"; [ -s "h$1.txt" ] || { echo "FAIL: no batch hashes ($D/tr$1.log)"; exit 1; }; }
run fly off; run st "$D/store"
grep -q 'dense store .* rendered' trst.log || { echo "FAIL: the store was not opened ($D/trst.log)"; exit 1; }
grep -qE 'from the dense store, [0-9.]+ s\), 0 computed' trst.log || { echo "FAIL: the store did not serve every cell ($D/trst.log)"; grep 'sampler-only: derived' trst.log; exit 1; }
cmp -s hfly.txt hst.txt || { echo "FAIL: batches differ with the store ($D/hfly.txt vs hst.txt)"; diff hfly.txt hst.txt | head -4; exit 1; }
# a partial store: every shard of every array marked unrendered (the header kept): computed, the same batches
for b in store/*/shards.bin; do head -c 36 "$b" > "$b.new"; n=$(( $(stat -c %s "$b") - 36 )); head -c $n /dev/zero >> "$b.new"; mv "$b.new" "$b"; done
run part "$D/store"; cmp -s hfly.txt hpart.txt || { echo "FAIL: batches differ with a partial store"; exit 1; }
echo "PASS: denselab store: $(grep -c byte-identical v.log) arrays byte-identical to a fresh compute, $(wc -l < hst.txt) sampler batches equal with the store, on the fly and a partial store"

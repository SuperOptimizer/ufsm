#!/usr/bin/env bash
# The universal-net recipe data parallel on two GPUs (train --gpus 0,1): latent grid, --rv 2, --um-rungs, --film 2, --fstem, --uw, --grad-cos,
# --gpu-aug 3 --replay 3 on the wind fixture's two sources (phase at levels 1 and 2), 6 steps, then a resume to 9. Then --det 1 for 30 steps:
# one host thread per GPU (the default) against one thread (UFSM_MGPU_THREADS=0, phased launches) and the per-GPU order (UFSM_MGPU_PHASED=0),
# step / validation / grad-cos lines identical (timing stripped; a race in the thread-local op state shows as a changed line)
# Usage: test_univ2.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3   # (absolute: the runs cd into DIR)
rm -rf "$D"; mkdir -p "$D"
"$MF" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1
python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
a = json.load(open(f'{d}/fx/sources-l1.json'))['sources'][0]; b = json.load(open(f'{d}/fx/sources2.json'))['sources'][0]; b.pop('holdout', None)
json.dump({'sources': [a, b]}, open(f'{d}/fx/sources-mix.json', 'w'))
PY
C=(--task unwrap2 --P 64 --B 1 --gpus 0,1 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --gpu-aug 3 --replay 3 --warmup 0 --seed 5 --widths 16,32,64 --rv 2 --rv-iso 100 --lat 4 --lat-guide 3 --um-rungs 19.2,38.4 --rung-shift-p 0.2
   --film 2 --fstem 16 --fstem-drop 0.25 --uw 1 --grad-cos 3 --packed-weight 8)
cd "$D"
"$U" train fx/sources-mix.json --out run --steps 6 "${C[@]}" > run.log 2>&1 || { echo "FAIL: 2-GPU run ($D/run.log)"; exit 1; }
grep -q 'x 2 GPU(s)' run.log || { echo "FAIL: not two GPUs"; exit 1; }
grep -q '^grad-cos step 6 (head input): phase/normal' run.log || { echo "FAIL: no grad-cos line"; exit 1; }
grep -q '^step      6 .* uw phase' run.log || { echo "FAIL: step 6 / uw"; exit 1; }
head -1 run/last.ckpt | grep -q '"film":2' && head -1 run/last.ckpt | grep -q '"fstem":\[16,' || { echo "FAIL: header"; exit 1; }
"$U" train fx/sources-mix.json --out run --steps 9 --resume run/last.ckpt "${C[@]}" > run2.log 2>&1 && grep -q '^step      9 ' run2.log || { echo "FAIL: resume ($D/run2.log)"; exit 1; }
CD=("${C[@]}" --det 1 --val-every 4)
for m in T:1:1 P:0:1 S:0:0; do IFS=: read n t p <<< "$m"
  UFSM_MGPU_THREADS=$t UFSM_MGPU_PHASED=$p "$U" train fx/sources-mix.json --out det$n --steps 30 "${CD[@]}" > det$n.log 2>&1 || { echo "FAIL: --det run $n ($D/det$n.log)"; exit 1; }
  grep -E '^step|^val|^grad-cos' det$n.log | sed -E 's/ [0-9.]+ samp\/s.*//' > det$n.txt
done
[ "$(grep -c '^step' detT.txt)" = 30 ] || { echo "FAIL: --det threads: $(grep -c '^step' detT.txt) step lines"; exit 1; }
for k in P S; do cmp -s detT.txt det$k.txt || { echo "FAIL: per-GPU threads differ from $k: $(cmp detT.txt det$k.txt)"; diff detT.txt det$k.txt | head -4; exit 1; }; done
echo "test_univ2: ok"

#!/bin/bash
# Single-GPU 512^3 confirmation, then held-out scoring against a baseline run.
# usage: tools/confirm_run.sh <name> <baseline-eval-dir> [extra ufsm train flags...]
# e.g.  tools/confirm_run.sh r11 /vesuvius/ufsm/eval/r10 --policy all=fp4:fp8:fp8,enc0.c1=fp16 --sr 1
set -euo pipefail
name=$1; base=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p runs/$name
train_gpus=${CGPUS:-0}; eval_gpu=${CEVAL_GPU:-${train_gpus%%,*}}
./build/ufsm train "${CCFG:-configs/all2.json}" --out "runs/$name" "$@" --gpus "${CGPUS:-0}" --P "${CP:-512}" --B "${CB:-1}" --steps "${CSTEPS:-1000}" --lr 1e-3 --warmup "${CWARM:-50}" --sched wsd --cooldown 0.2 \
  --soft 3 --intonly 1 --down-norm 1 --opt muon --muon-lr 0.01 --workers 32 --val-batches 8 --log-every 100 --val-every 200 --ckpt-every 1000 > "runs/$name.train.log" 2>&1
echo "$name: $(grep 'done in' "runs/$name.train.log") skips $(grep -c non-finite "runs/$name.train.log" || true)"
python3 tools/eval_holdouts.py "runs/$name/last.ckpt" --sources "${CCFG:-configs/all2.json}" --out "/vesuvius/ufsm/eval/$name" --gpu "$eval_gpu" --level 1 --scores "runs/$name.scores.json" > "runs/$name.eval.log" 2>&1
python3 tools/eval_holdouts.py "runs/$(basename "$base")/last.ckpt" --sources "${CCFG:-configs/all2.json}" --out "$base" --score-only --level 1 --scores "runs/$name.baseline.json" > "runs/$name.baseline.log" 2>&1
# Compare the same configured boxes at threshold 0.5, using only complete numeric results.
python3 - "$name" "$base" <<'PY'
import sys, json
name, base = sys.argv[1], sys.argv[2]
def read(path):
    return {k: next(r['f1'] for r in v['rows'] if r['threshold'] == 0.5)
            for k, v in json.load(open(path))['scores'].items()}
new, old = read(f'runs/{name}.scores.json'), read(f'runs/{name}.baseline.json')
assert new.keys() == old.keys(), 'baseline boxes differ'
ds = []
with open(f'runs/{name}.scores', 'w') as f:
    for k, v in new.items(): f.write(f'{k} {v:.4f}\n')
for k, v in new.items():
    o = old[k]; d = v - o; ds.append(d)
    print(f'{k:36s} {v:8.4f} {o:8.4f} {d:+.4f}')
if ds: print(f'mean dF1 {sum(ds)/len(ds):+.4f}  min {min(ds):+.4f}  n {len(ds)}')
PY

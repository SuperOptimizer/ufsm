#!/bin/bash
# 40k-step confirmation run on the full source set, then held-out scoring on every box against a baseline run.
# usage: tools/confirm_run.sh <name> <baseline-eval-dir> [extra ufsm train flags...]
# e.g.  tools/confirm_run.sh r11 /vesuvius/ufsm/eval/r10 --policy all=fp4:fp8:fp8,enc0.c1=fp16 --sr 1
set -u
name=$1; base=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p runs/$name
./build/ufsm train ${CCFG:-configs/all2.json} --out runs/$name --gpus 0,1 --P ${CP:-64} --B ${CB:-4} --steps ${CSTEPS:-40000} --lr 1e-3 --warmup 500 --sched wsd --cooldown 0.2 \
  --soft 3 --intonly 1 --down-norm 1 --opt muon --muon-lr 0.01 --workers 32 --val-batches 8 --log-every 100 --val-every 2000 --ckpt-every 10000 "$@" > runs/$name.train.log 2>&1
echo "$name: $(grep 'done in' runs/$name.train.log) skips $(grep -c non-finite runs/$name.train.log) best val $(grep -v '^step' runs/$name/log.csv | awk -F, '$8!=""{print $1,$8}' | sort -k2 -g | head -1)"
python3 tools/eval_holdouts.py runs/$name/best.ckpt --sources ${CCFG:-configs/all2.json} --out /vesuvius/ufsm/eval/$name --gpu 1 --level 1 2>&1 | grep -E "^==|^0.50" | paste - - | awk '{print $2, $8}' > runs/$name.scores
# per-box comparison with the baseline (F1 at 0.5) and the mean difference
python3 - "$name" "$base" <<'PY'
import sys, os, glob, subprocess
name, base = sys.argv[1], sys.argv[2]
new = dict(l.split() for l in open(f'runs/{name}.scores') if len(l.split()) == 2)
# baseline scores: rebuild from its eval dir with the same scorer if no .scores file
bs = f'runs/{os.path.basename(base)}.scores'
old = dict(l.split() for l in open(bs) if len(l.split()) == 2) if os.path.exists(bs) else {}
ds = []
for k, v in new.items():
    o = old.get(k)
    d = (float(v) - float(o)) if o else None
    if d is not None: ds.append(d)
    print(f'{k:36s} {v:>8s} {o or "-":>8s} {"" if d is None else f"{d:+.4f}"}')
if ds: print(f'mean dF1 {sum(ds)/len(ds):+.4f}  min {min(ds):+.4f}  n {len(ds)}')
PY

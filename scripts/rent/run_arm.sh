#!/usr/bin/env bash
# scripts/rent/run_arm.sh NAME [GPU]: one pure-SSL arm (the multi-sample SSL batch; every supervised term 0; the batch-1 main window minimal)
# over the 30 native ~8.6-9.4 um U0 scans (sources-rent.json: CT streamed from dl.ash2txt.org into ./cache, 12 papyrus boxes of 384^3 per
# scan from the scanprep masks; no labels). The arm's options: the NAME row of sweep.tsv. Output: runs/NAME/ (train.log, checkpoints every
# 2000 steps with their .ssl heads) and runs/NAME.log. Detached; STOP: touch runs/NAME/STOP. Several arms on one box: one per GPU.
set -eu
N=${1:?arm name (sweep.tsv)}; G=${2:-0}; K=$(cd "$(dirname "$0")" && pwd); U=${U:-$K/../../build/ufsm}
A=$(awk -F'\t' -v n="$N" '$1 == n {print $2}' "$K/sweep.tsv"); [ -n "$A" ] || { echo "run_arm: no arm $N in sweep.tsv"; exit 2; }
mkdir -p runs/$N cache
setsid nohup "$U" dino-train --gpu "$G" $A --sources "$K/sources-rent.json" --out runs/$N --recipe 1 --P 64 --source-P 96 --local-P 32 --locals 1 \
  --attn-blocks 2 --attn-reg 1 --cls both --head-slab 16 --mem lean --linear-mx 4 --protos-k 1024 --ibot-k 1024 \
  --gt 0 --object 0 --region 0 --ink 0 --dino 0 --ibot 0 --koleo 0 --var 0 --proto-lr 0 --workers 2 --threads 2 \
  --log-every 50 --save-every 2000 --keep-every 2000 --steps 1000000 --eval-every 100000000 >> runs/$N.log 2>&1 < /dev/null &
echo "run_arm: $N on GPU $G (pid $!): tail -f runs/$N.log"

#!/bin/bash
# train.sh NAME STEPS [extra] (CAL, POD-L): v2 density calibration, wind96 warm start (COUNT's dft/train.sh recipe) on multi-scroll published GT
# (sources-cal.json: P4 June + 0172 + 0814 + 0500P2, balanced per scroll; 0139 / 1667 never), --density-packed 0 --count-lambda CL (0.05)
# --count-strata, trunk LR x0.25 for the first 2k steps, ckpt every 500. Env: CL, P (512), STRATA, MLR, ALR, SRC. Stop: kill -TERM $(cat runs/NAME.pid) after a checkpoint.
set -u; N=$1; S=$2; shift 2; C=/root/cal; U=$C/src/build/ufsm; o=$C/runs/$N; mkdir -p $C/runs
AUG="--zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20 \
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0 \
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-hist-p 0.15 --aug-hist $C/misc/target-hist.txt --aug-noise-p 0.15 --aug-cnoise-p 0.1 \
  --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1"
R=(--init-from /root/pod/ckpt/wind96-wk1-best.ckpt); [ -f $o/last.ckpt ] && R=(--resume $o/last.ckpt)
export MALLOC_ARENA_MAX=4
echo $$ > $o.pid
nice -n 5 $U train ${SRC:-$C/sources-cal.json} --out $o --gpus 0 --P ${P:-512} --B 1 \
  --widths 32,64,96,128,160,192 --dec-widths 0,32 --scale-cond 1 --task unwrap2 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1 \
  --density-packed 0 --count-lambda ${CL:-0.05} --count-segments 1024 --count-length 64 --count-strata ${STRATA:-70,115,130,230} \
  --trunk-lr-mult 0.25 --trunk-lr-steps 2000 \
  --steps $S --sched wsd --cooldown 0.2 --opt muon --fp4 2 --gpu-aug 2 --input-prec 8 --mem auto --val-store disk \
  --workers ${WK:-10} --sampler-slots 16 --log-every 50 --val-every 500 --val-batches 16 --ckpt-every 500 --seed 3 --val-seed 1 \
  --warmup 200 --muon-lr ${MLR:-0.003} --lr ${ALR:-0.0002} $AUG "${R[@]}" "$@" >> $o.log 2>&1

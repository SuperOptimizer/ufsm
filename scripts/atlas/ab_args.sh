# atlas A/B train args (sourced): try9cool's flags exactly (trainperf/admuon2/try9.argv) except: sources $SRCJ, --out $OUT, --steps $STEPS,
# --init-from try9 cooled-33333 (no --resume), --init-lr-mult 0.3 over the whole run, val every $VAL, device flags from the env (CPU passes: -1).
U=${U:-/vesuvius/ufsm/bin/ufsm-atlas}
INIT=${INIT:-/vesuvius/ufsm/v2/lat/runs/try9cool/cooled-33333.ckpt}; P=${P:-768}
A=($SRCJ --out $OUT --gpus 0 --P $P --B 1 --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1
  --fibre-gpu ${FIBGPU:-1} --ink-gpu ${INKGPU:-1} --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1
  --steps ${STEPS:-100} --sched wsd --cooldown 0.1 --opt muon --fp4 2 --gpu-aug ${GPUAUG:-3} --mem auto --val-store regen --workers ${WORKERS:-10} --sampler-slots ${SLOTS:-12}
  --log-every 10 --val-every ${VAL:-1000000} --val-batches ${VALB:-32} --ckpt-every 50 --seed ${SEED:-3} --val-seed 1 --warmup 0 --muon-lr 0.003 --lr 0.0003
  --zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0 --aug-tone-p 0.3
  --aug-gin-p 0.15 --aug-noise-p 0.15 --aug-cnoise-p 0.1 --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1
  --lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --um-rungs 4.8,9.6:2,19.2 --rung-shift-p 0.3
  --film 2 --fstem 16 --fstem-drop 0.25 --uw 0 --grad-cos 500 --ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1 --ink-cache-mb 4096
  --mix-every 250 --onfly-slots ${ONFLYSLOTS:-3} --onfly-mb ${ONFLYMB:-384} --strict-sources 1 --muon-tc 2 --normuon 1 --val-gn-live 1 --gn-ema-warm 1000000000
  --skip-empty 1 --skip-thr 0 --skip-margin 1 --echo-max ${ECHOMAX:-4} --accum ${ACCUM:-10} --holdout-guard 32 --stick-src ${STICK:-8} --snap-jitter 1 --edge-boost 1 --vox-cap 2 --stop-at-update 1
  --init-from $INIT --init-lr-mult 0.3 --init-lr-steps 1000000 --ct-gpu-decode ${CTGD:-1} --onfly-gpu ${ONFLYGPU:-1} --ema 0.9995 --ema-per window
  --snap-every 1000000 ${EXTRA:-})

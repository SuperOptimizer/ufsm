#!/usr/bin/env bash
# Measured performance baseline for the production recipe: build + make test, a short P704 split-z training profile
# (per-op and per-layer GPU time), and a 1024^3 two-GPU inference benchmark with the serving settings and the
# predict stage profile. Needs idle GPUs; refuses to start while a ufsm process is running.
# usage: tools/perf_baseline.sh <run-root> <out-dir>
set -euo pipefail
R=${1:?run root}; O=${2:?output dir}; mkdir -p "$O"
if pgrep -x ufsm >/dev/null; then echo "a ufsm process is running; GPUs are not idle" >&2; exit 1; fi
cd "$(dirname "$0")/.."
B=$PWD/build/ufsm
make -j"$(nproc)" >"$O/build.log" 2>&1
make test >"$O/test.log" 2>&1 && echo "make test: PASS" || echo "make test: FAIL (see $O/test.log)"
IN=$R/full/inputs
TR=(train "$IN/sources.json" --gpus 0,1 --fp4 2 --mem auto --opt muon --muon-lr 0.01 --lr 0.001 --sched wsd --cooldown 0.2
    --soft 2 --down-norm 1 --workers 12 --seed 2 --input-prec 8 --gn-stats stored --val-batches 2 --val-every 100000 --ckpt-every 100000 --log-every 20
    --levels 1,0,0,0 --zfix 1 --symmetry-p 0.5 --ct-aug 1 --axis-jitter 32 --det 1 --cover "$IN/cover.json"
    --widths 16,32,64,80 --split z --P 704 --B 1 --warmup 10 --erode 1)
# training: steady-state rate (no profiler), then per-op and per-layer profiles
UFSM_RC_KEEP_COARSE=1 "$B" "${TR[@]}" --steps 200 --out "$O/train_rate" >"$O/train_rate.log" 2>&1 || echo "train rate run failed"
UFSM_RC_KEEP_COARSE=1 UFSM_PROF=1 "$B" "${TR[@]}" --steps 60 --out "$O/train_prof" >"$O/train_prof.log" 2>&1 || echo "train prof run failed"
UFSM_RC_KEEP_COARSE=1 UFSM_PROF=layers "$B" "${TR[@]}" --steps 40 --out "$O/train_layers" >"$O/train_layers.log" 2>&1 || echo "train layers run failed"
grep "samp/s" "$O/train_rate.log" | tail -3
# inference: serving recipe on the 1024^3 held-out box, both GPUs; then one GPU with the stage profile
CK=$R/best-development.ckpt
PR=(predict "$CK" /vesuvius/usrm/volcomp PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr)
PO=(--um 2.4 --box 48128,17408,16384,1024,1024,1024 --halo 8 --shard 512 --grid-origin 0,0,0 --ema 1 --q 8 --threads 16
    --axis "$IN/axis-0.json" --levels 1)
for w in 720 528; do
  rm -rf "$O/pred$w"
  /usr/bin/time -f "%e s wall" "$B" "${PR[@]}" "$O/pred$w" "${PO[@]}" --window $w --gpus 0,1 >"$O/pred$w.log" 2>&1 || echo "predict $w failed"
  echo "predict window $w two GPUs: $(tail -1 "$O/pred$w.log")"
  rm -rf "$O/pred${w}p"
  UFSM_PRED_PROF=1 "$B" "${PR[@]}" "$O/pred${w}p" "${PO[@]}" --window $w --gpu 0 >"$O/pred${w}p.log" 2>&1 || echo "predict prof $w failed"
  grep "predict profile" "$O/pred${w}p.log" || true
done

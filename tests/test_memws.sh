#!/usr/bin/env bash
# Kernel workspaces inside the memory plan (2026-11-14 audit net#4): try6's flag set (the 92 M latent net, --fp4 2, unwrap2 + rv 2 + ink head /
# MIL + fibre head, fine stem, skip-empty, accum, echo) at P 256 on the wind fixture, 4 steps, UFSM_MEM_REPORT=4. FAIL if any kernel workspace
# (nn_ws) grows after the planner allocated and sealed them ("grew after the memory plan"), or if any device allocation of >= 1 MB
# ("alloc" lines of nn_malloc) happens after the first step line: everything a step needs exists after step 1.
# Usage: test_memws.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
case "$D" in */memws*) rm -rf -- "$D";; esac
mkdir -p "$D"
"$MF" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1
S=fx/sources-ink.json; [ -f "$D/$S" ] || S=fx/sources.json
C=(--gpus 0 --P 256 --B 1 --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1
   --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1 --steps 4 --sched wsd --cooldown 0.2
   --opt muon --fp4 2 --gpu-aug 3 --mem auto --workers 2 --sampler-slots 2 --log-every 1 --val-batches 0 --val-every 100000 --ckpt-every 100000 --seed 3
   --warmup 2 --muon-lr 0.006 --lr 0.0006 --zfix 1 --aug none --lat 4 --lat-guide 3 --film 2 --fstem 16 --fstem-drop 0.25 --uw 0 --muon-tc 2
   --skip-empty 1 --skip-thr 0 --skip-margin 1 --accum 2 --echo-max 2)
[ "$S" = fx/sources-ink.json ] && C+=(--ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1)
cd "$D"
UFSM_MEM_REPORT=4 "$U" train $S --out run "${C[@]}" > run.log 2>&1 || { echo "FAIL: run ($D/run.log)"; tail -5 run.log; exit 1; }
grep -q '^step      4 ' run.log || { echo "FAIL: no step 4 ($D/run.log)"; exit 1; }
grep -q '^memory: kernel workspaces .* allocated and sealed' run.log || { echo "FAIL: no workspace plan line ($D/run.log)"; exit 1; }
if grep -n 'grew after the memory plan' run.log; then echo "FAIL: a kernel workspace grew after the plan ($D/run.log)"; exit 1; fi
late=$(awk '/^step / {s = 1} s && /^alloc / {print}' run.log)
[ -z "$late" ] || { echo "FAIL: device allocations after step 1 ($D/run.log):"; echo "$late" | head -5; exit 1; }
echo "test_memws: ok"

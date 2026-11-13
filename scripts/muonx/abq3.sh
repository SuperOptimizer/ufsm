#!/bin/bash
# Muon-variant A/B queue (latres 2026-11-13, branch muonx-20261113): one worker per GPU claims arms from arms.txt in order (mkdir claim).
# The 92 M net (latdino's LR-probe recipe), P 512 B 1, --det 1 --seed 7 (the same windows in every arm), Muon 0.01 / AdamW 1e-3,
# warmup 10% of the steps, WSD cooldown 0.2. One arm per card (~11.2 GB at P 512).
# GPU 1: flock gpu1.lock (blocking: the trainer queue). GPU 0: starts an arm only when gpu0.lock and gpu0-bench.lock are both free and
# no inkB flock waiter exists (inkB has GPU 0 priority), holding both for the arm.
# arms.txt lines: NAME STEPS FLAGS...   usage: abq.sh GPU
D=/vesuvius/ufsm/v2/lat/runs/muonx; U=$D/ufsm.muonx; G=$1
waiters() { ps -eo stat=,comm=,args= | awk -v g="/tmp/gpu$G-bench.lock" '$1 !~ /^T/ && $2 == "flock" && index($0, g)' | wc -l; }   # (flock processes only, not stopped)
arm() {   # NAME STEPS FLAGS...: a flag also in the base recipe replaces its value there (opt() reads the first occurrence)
  local n=$1 st=$2; shift 2; local o=$D/runs/$n w=$(( st / 10 )); rm -rf $o; mkdir -p $D/runs
  local base=(--gpus 0 --P 512 --B 1 --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25
    --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1 --steps $st --warmup $w --sched wsd --cooldown 0.2 --opt muon --muon-lr 0.01 --lr 0.001
    --det 1 --seed 7 --workers 6 --sampler-slots 12 --mem auto --log-every 10 --ckpt-every 100000 --val-batches 0
    --lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --um-rungs 4.8,9.6:2,19.2 --film 2
    --fstem 16 --fstem-drop 0.25 --uw 1 --ink-head 1 --ink-mil 1 --ink-dice 0 --onfly-slots 1 --gn-ema 1 --scan-norm 1 --fib-head 1)
  local x=("$@") extra=() i j hit
  for ((i = 0; i < ${#x[@]}; i++)); do
    hit=0; if [ $((i + 1)) -lt ${#x[@]} ] && [ "${x[i+1]:0:2}" != -- ]; then for ((j = 0; j < ${#base[@]}; j++)); do [ "${base[j]}" = "${x[i]}" ] && { base[j+1]=${x[i+1]}; hit=1; }; done; fi
    if [ $hit = 1 ]; then i=$((i + 1)); else extra+=("${x[i]}"); fi
  done
  env CUDA_VISIBLE_DEVICES=$G ~/bin/memcap -t nice -n 5 $U train /vesuvius/ufsm/univ/scratch1/sources-scratch1.json --out $o "${base[@]}" "${extra[@]}" > $D/runs/$n.log 2>&1
  echo "$(date +%F_%T) gpu$G $n rc=$?" >> $D/done.log
}
next() {   # claim the first unclaimed arm (arms.txt re-read each time: arms added later are picked up); prints "NAME STEPS FLAGS"
  while read -r n st fl; do
    [ -z "$n" ] || [ "${n:0:1}" = "#" ] && continue
    mkdir $D/claim.$n 2>/dev/null && { echo "$n $st $fl"; return 0; }
  done < $D/arms.txt; return 1
}
run1() { local a; a=$(next) || return 1; set -- $a; echo "$(date +%F_%T) gpu$G start $1" >> $D/done.log; arm "$@"; }
# between arms: hold off while $D/pause.gpuG exists or another job waits on this GPU's lock (univI / inkB short tests go first)
others() { ps -eo stat=,comm=,args= | awk -v a="/vesuvius/ufsm/opt/gpu$G.lock" -v b="/tmp/gpu$G-bench.lock" '$1 !~ /^T/ && $2 == "flock" && (index($0, a) || index($0, b))' | wc -l; }
while :; do
  while [ -e $D/pause.gpu$G ] || [ "$(others)" != 0 ]; do sleep 30; done
  exec 8>/vesuvius/ufsm/opt/gpu$G.lock 9>/tmp/gpu$G-bench.lock
  if flock -n 8 && { [ "$G" = 1 ] || flock -n 9; }; then run1; r=$?; flock -u 9; flock -u 8; exec 8>&- 9>&-; [ $r = 0 ] || break
  else exec 8>&- 9>&-; sleep 30; fi
done

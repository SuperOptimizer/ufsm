#!/bin/bash
# sweepB.sh RUN...: the compression sweep's training part on the pod (DESIGN 2026-11-12). RUN = ARM[:SEED] (seed 3 by default):
#   q2            the control: the mirror's CT as it is            vqQ: the training CT re-encoded at volcomp quality Q
#   lvC_S         the training CT as latent cells (C coefficients, step S x sd)
# Each run: U2's recipe (/workspace/univA/u2.args: --accum, P 1024) without the ink head, from u1-10000, STEPS (1000) windows, sources =
# Paris 4 June / July + PHerc1667-gt (the u1 pod kit's entry) at the 9.6 um rung with the CT root swapped; then the 14 wbench boxes scored twice, on the uncompressed anchor
# (sw/orig) and on the run's own codec (sw/ARM, re-encoded from the anchor). Training CT: the mirror's level 2 re-encoded whole
# (b/ct/ARM). The trainer is a shared GPU job that yields to gpu.pause: on a pause it is stopped with SIGTERM (a graceful save) and
# resumed when the pause is gone. Rows: wb/res/B<run>-{orig,own}.tsv; logs b/log.
L=/workspace/latstore; H=$L/ufsmB; U=$H/build/ufsm; BB=$L/b; SW=$L/sw; K=/workspace/ue; mkdir -p $BB/log $BB/ct $BB/runs
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; P4=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
P7=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr
STEPS=${STEPS:-1000}; BX=${BX:-PH1 PH2 PH3 HOLD J1 J9 J19 J27 C9 C18 C30 C52 139-5 139-2}
for run in "$@"; do a=${run%%:*}; seed=3; [ "$run" != "$a" ] && seed=${run##*:}; r=$a-s$seed; o=$BB/runs/$r
  # 1. the training CT
  root=$M
  if [ $a != q2 ]; then root=$BB/ct/$a
    case $a in vq*) args="--volcomp ${a#vq}";; lv*) c=${a#lv}; args="--c ${c%_*} --step ${c#*_} --gpu -1";; esac
    for key in $P4 $P7; do [ -e $root/$key/.done ] && continue; rm -rf $root/$key; mkdir -p $root/$(dirname $key)
      nice -n 10 $U latvol $L/klt1.lcb $M $key $root/$key --levels 2 --threads ${ETH:-20} --cache /workspace/univ/cache $args > $BB/log/enc_${a}_$(basename $(dirname $(dirname $key))).log 2>&1 || { echo "encode $a $key failed"; continue 2; }
      touch $root/$key/.done; tr '\r' '\n' < $BB/log/enc_${a}_$(basename $(dirname $(dirname $key))).log | grep -aE '^rate:|GB' | tail -2 | cut -c1-300; done; fi
  # 2. the sources: Paris 4 June / July at level 2 from that root
  python3 -I - $root $BB/src-$a.json <<'PY'
import json, sys
S = json.load(open('/workspace/univA/sources-u2pod.json')); out = []
for s in S['sources']:
    if s['name'] in ('PHercParis4-june', 'PHercParis4-july'): s['root'] = sys.argv[1]; s['min_level'] = s['max_level'] = 2; out.append(s)
S1 = json.load(open('/workspace/univ/sources-u1-pod.json'))
for s in (S1['sources'] if isinstance(S1, dict) else S1):
    if s['name'] == 'PHerc1667-gt': s['root'] = sys.argv[1]; out.append(s)
S['sources'] = out; json.dump(S, open(sys.argv[2], 'w'), indent=1)
PY
  # 3. the finetune (yields to gpu.pause)
  if ! grep -q "^done" $o.state 2>/dev/null; then
    read -r -a A < /workspace/univA/u2.args; ARGS=($BB/src-$a.json); i=1
    while [ $i -lt ${#A[@]} ]; do k=${A[$i]}; v=${A[$((i + 1))]:-}
      case $k in --out) v=$o;; --steps) v=$STEPS;; --seed) v=$seed;; --warmup) v=40;; --um-rungs) v=9.6;; --ckpt-every) v=250;; --val-every) v=100000;; --workers) v=${WORKERS:-10};;
        --ink-head|--ink-mil|--ink-dice|--ink-smooth) i=$((i + 2)); continue;; esac
      ARGS+=($k $v); i=$((i + 2)); done
    while :; do
      while [ -e $K/gpu.pause ]; do sleep 5; done
      RS=(); [ -f $o/last.ckpt ] && RS=(--resume $o/last.ckpt)
      mkdir -p $K/gpu.shared; touch $K/gpu.shared/latstore-B
      UFSM_LATVOL_GPU=${LVGPU-} nice -n 5 $U train "${ARGS[@]}" "${RS[@]}" >> $o.log 2>&1 & pid=$!
      while kill -0 $pid 2>/dev/null; do [ -e $K/gpu.pause ] && { kill -TERM $pid; echo "$(date +%T) yielding to $(cat $K/gpu.pause 2>/dev/null)" >> $o.state; wait $pid; break; }; sleep 3; done
      wait $pid; rc=$?; rm -f $K/gpu.shared/latstore-B
      last=$(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $o.log)
      [ "$last" -ge "$STEPS" ] && { echo "done $last" >> $o.state; break; }
      [ $rc != 0 ] && [ ! -e $K/gpu.pause ] && { n=$((${n:-0} + 1)); [ $n -ge 3 ] && { echo "run $r failed rc $rc (see $o.log)"; break; }; sleep 20; }
    done
  fi
  grep -q "^done" $o.state 2>/dev/null || continue
  # 4. the scores: the anchor CT and the run's own codec
  own=$a; [ $a = q2 ] && own=vq2
  [ -e $SW/$own/$P4/.done ] || ENCONLY=1 BX="$BX" bash $L/ufsm/scripts/latvol/sweep.sh $own > $BB/log/encbox_$own.log 2>&1
  ( UPBIN=$U CK=$o/last.ckpt TAG=B$r LATGRID=1 LVBASE=$SW THREADS=10 bash $L/ufsm/scripts/latvol/pod_wb.sh orig $BX > $BB/log/wb_${r}_orig.log 2>&1 ) &
  ( UPBIN=$U CK=$o/last.ckpt TAG=B$r LATGRID=1 LVBASE=$SW THREADS=10 bash $L/ufsm/scripts/latvol/pod_wb.sh $own $BX > $BB/log/wb_${r}_own.log 2>&1 ) &
  wait; echo "run $r done $(date +%T)"
  # the arm's training CT goes once no later run of this call needs it (the pod's disk)
  case " $* " in *" $a:"*) [ "$run" = "$a" ] && continue;; esac
  [ $a != q2 ] && [ -n "$a" ] && [ "${KEEPCT:-0}" != 1 ] && rm -rf "$BB/ct/$a"
done

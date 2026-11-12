#!/bin/bash
# sweepB.sh RUN...: the compression sweep's training part on the pod (DESIGN 2026-11-12). RUN = ARM[:SEED] (seed 3 by default):
#   q2            the control: the mirror's CT as it is            vqQ: the training CT re-encoded at volcomp quality Q
#   lvC_S         the training CT as latent cells (C coefficients, step S x sd)
# Each run: U2's recipe (/workspace/univA/u2.args: --accum, P 1024) without the ink head, from u1-10000, STEPS (500) windows, sources =
# Paris 4 June / July at the 9.6 um rung (on-the-fly phase) with the CT root swapped. A small shared job beside U2 (the lead's budget:
# planner pinned to 14 GB of the card, 2 workers / 2 sampler slots, one on-the-fly slot; host RSS watched, killed above RSSMAX 35 GB).
# It yields to gpu.pause (SIGTERM = a graceful save, resumed when the pause is gone); while a run trains, /workspace/latstore/b.busy
# holds my other shared units (pod_shared.sh). Then the Paris 4 clean boxes are scored twice with the run's checkpoint: fed the q2
# mirror, and fed the run's own codec (box regions re-encoded from the mirror, b/box/ARM). Training CT: the mirror's level 2 re-encoded
# whole (b/ct/ARM, removed after the run). Rows: wb/res/B<run>-{ct,ARM}.tsv; logs b/log.
L=/workspace/latstore; H=$L/ufsmB; U=$H/build/ufsm; BB=$L/b; K=/workspace/ue; mkdir -p $BB/log $BB/ct $BB/box $BB/runs
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; P4=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
STEPS=${STEPS:-300}; RSSMAX=${RSSMAX:-35}; CTCACHE=${CTCACHE:-/workspace/univ/cache}
BX=${BX:-PH1 PH2 PH3 HOLD J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 J11 J12 J13 J14 J15 J16 J17}
for run in "$@"; do a=${run%%:*}; seed=3; [ "$run" != "$a" ] && seed=${run##*:}; r=$a-s$seed; o=$BB/runs/$r
  case $a in q2) args=;; vq*) args="--volcomp ${a#vq}";; lv*) c=${a#lv}; args="--c ${c%_*} --step ${c#*_} --gpu -1";; esac
  # 1. the training CT
  root=$M
  if [ $a != q2 ] && ! grep -q "^done" $o.state 2>/dev/null; then root=$BB/ct/$a
    if [ ! -e $root/$P4/.done ]; then rm -rf "$BB/ct/$a"; mkdir -p $root/$(dirname $P4)
      nice -n 10 $U latvol $L/klt1.lcb $M $P4 $root/$P4 --levels 2 --threads ${ETH:-16} --cache $CTCACHE $args > $BB/log/enc_$a.log 2>&1 || { echo "encode $a failed"; continue; }
      touch $root/$P4/.done; tr '\r' '\n' < $BB/log/enc_$a.log | grep -aE 'GB' | tail -1 | cut -c1-300; fi; fi
  # 2. the sources: Paris 4 June / July at level 2 from that root
  python3 -I - $root $BB/src-$a.json $BB/cache-$a <<'PY'
import json, sys
S = json.load(open('/workspace/univA/sources-u2pod.json')); out = []
for s in S['sources']:
    if s['name'] in ('PHercParis4-june', 'PHercParis4-july'): s['root'] = sys.argv[1]; s['min_level'] = s['max_level'] = 2; out.append(s)
# the chunk cache: its path has no root in it (zarr3.c cache_path), so a re-encoded arm must never share the kit's cache with the mirror.
# An arm's root is a local tree (z3_open gives local stores no cache at all); on top of that its sources name a cache directory of
# its own, so the kit's cache path is not in the arm's inputs. Only the q2 control (the mirror itself) keeps the kit's cache.
if not sys.argv[1].startswith('http'): S['cache'] = sys.argv[3]
S['sources'] = out; json.dump(S, open(sys.argv[2], 'w'), indent=1)
print('sources %s: root %s, cache %s' % (sys.argv[2], sys.argv[1], S.get('cache')))
PY
  [ $a != q2 ] && { mkdir -p $BB/cache-$a; grep -c '/workspace/univ/cache' $BB/src-$a.json | sed "s|^|arm $a: references to the kit's chunk cache in its sources: |"; }
  # 3. the finetune
  if ! grep -q "^done" $o.state 2>/dev/null; then
    read -r -a A < /workspace/univA/u2.args; ARGS=($BB/src-$a.json); i=1
    while [ $i -lt ${#A[@]} ]; do k=${A[$i]}; v=${A[$((i + 1))]:-}
      case $k in --out) v=$o;; --steps) v=$STEPS;; --seed) v=$seed;; --warmup) v=20;; --um-rungs) v=9.6;; --ckpt-every) v=50;; --val-every) v=100000;; --workers) v=2;; --sampler-slots) v=2;;
        --onfly-mb) v=4096;; --log-every) v=10;; --ink-head|--ink-mil|--ink-dice|--ink-smooth|--onfly-slots) i=$((i + 2)); continue;; esac
      ARGS+=($k $v); i=$((i + 2)); done
    ARGS+=(--onfly-slots 1); nf=0
    while :; do
      # shared jobs own minutes :00-:20 of the hour: a segment starts before :12 and is stopped (SIGTERM: a save) at :19
      while [ -e $K/gpu.pause ] || [ $((10#$(date +%M))) -ge ${SEGSTART:-12} ] || pgrep -f "[s]weep_ink.sh" > /dev/null; do sleep 5; done
      touch $L/b.busy; while [ -n "$(ls $K/gpu.shared 2>/dev/null | grep '^latstore-')" ]; do sleep 5; done   # my other units drain first
      # U2 needs its ~64 GB: start only with >= FREEGB (20) GB free on the card (this job plans 14 + one 4 GB on-the-fly slot)
      while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt $((${FREEGB:-20} * 1024)) ]; do sleep 20; [ -e $K/gpu.pause ] && break; done
      [ -e $K/gpu.pause ] && { rm -f $L/b.busy; continue; }
      RS=(); [ -f $o/last.ckpt ] && RS=(--resume $o/last.ckpt)
      mkdir -p $K/gpu.shared; touch $K/gpu.shared/latstore-B
      UFSM_MEM_FREE_GB=14 UFSM_HEAD_SLAB_Z=16 nice -n 5 $U train "${ARGS[@]}" "${RS[@]}" >> $o.log 2>&1 & pid=$!; why=
      while kill -0 $pid 2>/dev/null; do
        rss=$(awk '/VmRSS/ {print int($2 / 1048576)}' /proc/$pid/status 2>/dev/null); echo "${rss:-0}" > $o.rss_gb
        [ "${rss:-0}" -gt $RSSMAX ] && { kill -KILL $pid; why="rss $rss GB > $RSSMAX"; break; }
        [ -e $K/gpu.pause ] && { kill -TERM $pid; why="yield to $(cat $K/gpu.pause 2>/dev/null)"; break; }
        [ $((10#$(date +%M))) -ge ${SEGEND:-19} ] && { kill -TERM $pid; why="yield: the window ends (step $(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $o.log))"; break; }; sleep 3; done
      wait $pid; rc=$?; rm -f $K/gpu.shared/latstore-B $L/b.busy; [ -n "$why" ] && echo "$(date +%T) $why" >> $o.state
      last=$(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $o.log)
      [ "$last" -ge "$STEPS" ] && { echo "done $last" >> $o.state; break; }
      case "$why" in rss*) echo "run $r killed: $why"; break;; yield*) continue;; esac
      nf=$((nf + 1)); [ $nf -ge 2 ] && { echo "run $r failed rc $rc (see $o.log)"; break; }; sleep 20
    done
  fi
  [ $a != q2 ] && echo "arm $a: files in its own cache dir after the run: $(find $BB/cache-$a -type f 2>/dev/null | wc -l) (a local root uses none)"
  [ $a != q2 ] && [ -n "$a" ] && [ "${KEEPCT:-0}" != 1 ] && rm -rf "$BB/ct/$a"
  grep -q "^done" $o.state 2>/dev/null || continue
  # 4. the scores: fed the q2 mirror, and fed the run's own codec (the boxes' regions re-encoded from the mirror)
  ( UPBIN=$U CK=$o/last.ckpt TAG=B$r LATGRID=1 THREADS=8 bash $L/ufsm/scripts/latvol/pod_wb.sh ct $BX > $BB/log/wb_${r}_ct.log 2>&1 ) &
  if [ $a != q2 ]; then
    if [ ! -e $BB/box/$a/$P4/.done ]; then bash $L/ufsm/scripts/latvol/wb_boxes.sh paris4 $BX > $BB/log/boxes; rm -rf "$BB/box/$a"; mkdir -p $BB/box/$a/$(dirname $P4)
      nice $U latvol $L/klt1.lcb $M $P4 $BB/box/$a/$P4 --levels 2 --threads 12 --cache $K/cache --boxes $BB/log/boxes $args > $BB/log/encbox_$a.log 2>&1 && touch $BB/box/$a/$P4/.done; fi
    ( UPBIN=$U CK=$o/last.ckpt TAG=B$r LATGRID=1 LVBASE=$BB/box THREADS=8 bash $L/ufsm/scripts/latvol/pod_wb.sh $a $BX > $BB/log/wb_${r}_$a.log 2>&1 ) &
  fi
  wait; echo "run $r done $(date +%T): $(grep -c . $L/wb/res/B$r-ct.tsv) rows, $(grep -E '^step +[0-9]+ ' $o.log | tail -1 | cut -c1-120)"
done

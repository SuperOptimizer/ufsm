#!/usr/bin/env bash
# desksmoke.sh: inksdk's latent-ink smoke on the desk (lead 11-18: no pods): card 1 only, chained after unrollperf's 1667 unroll
# (waits for /vesuvius/ufsm/unrollperf/q2.done), arm A from u1-13000-w on PHerc1667 / 0814 / MAN5 / 0009B (sources from mkarms.py
# SCROLLS, CT prefetched into /vesuvius/ufsm/cache beforehand), STEPS 200, the step-10 / step-50 gates and the step-200 sanity of
# scripts/inksmoke/podq.sh, a wall cap; then /vesuvius/ufsm/inksdk/smoke.done. Locks: suite gpu1.excl, then opt/gpu1.lock, each
# taken with flock -x -w 5 and released again if the other is not free (back-off 60 s); u1.train.sh gets LOCK="" (held here).
# Run: setsid nohup ~/bin/memcap -t -m 30G bash desksmoke.sh > desksmoke.out 2>&1 < /dev/null &
set -uo pipefail
T=/vesuvius/ufsm/inksdk/smoke; L=$T/smoke.log; RD=$T/runs; O=$T/out; DONE=/vesuvius/ufsm/inksdk/smoke.done; mkdir -p $RD $O
STEPS=${STEPS:-200}; ARMMAX=${ARMMAX:-2700}; MINSPS10=${MINSPS10:-0.03}; MINSPS=${MINSPS:-0.1}; MINFREE=${MINFREE:-11000}
U=$T/ufsm; D=/home/forrest/ufsm-inksdk/scripts/inksmoke; INIT=/vesuvius/ufsm/podun/u1-13000-w.ckpt; LATB=/vesuvius/ufsm/inksmoke/kit/klt1.lcb
say() { echo "$(date '+%F %T') desksmoke: $*" | tee -a $L; }
fin() { say "$*"; cp $L $O/ 2>/dev/null; echo "$(date '+%F %T') $*" > $DONE; exit ${2:-0}; }
say "start; waiting for /vesuvius/ufsm/unrollperf/q2.done"
until [ -e /vesuvius/ufsm/unrollperf/q2.done ]; do sleep 30; done
say "q2.done seen"
PF=$(pgrep -u $USER -x ufsm -a 2>/dev/null | awk '/ prefetch .*inksdk\/smoke/ {print $1}' | head -1)
[ -n "$PF" ] && { say "prefetch $PF still running: waiting (the trainer reads the warm cache)"; while kill -0 $PF 2>/dev/null; do sleep 30; done; }
say "prefetch: $(tail -c 300 $T/prefetch.log | tr '\r' '\n' | tail -1)"
# locks: gpu1.excl (suite) first, then opt/gpu1.lock; never hold one -x while waiting on the other
while :; do
  exec 7> /tmp/ufsm-suite-locks/gpu1.excl
  if flock -x -w 5 7; then exec 8> /vesuvius/ufsm/opt/gpu1.lock; if flock -x -w 5 8; then break; fi; exec 8>&-; fi
  exec 7>&-; sleep 60
done
say "holding gpu1.excl + opt/gpu1.lock"
fr=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
say "card 1 free ${fr} MiB (try9's share there 1.8-3.7 GB)"
t0=$(date +%s)
( CUDA_VISIBLE_DEVICES=1 R=$RD U=$U INIT=$INIT LATB=$LATB GPU=0 LOCK="" RELAUNCH=1 WORKERS=${WORKERS:-1} SLOTS=${SLOTS:-2} STEPS=$STEPS SNAP=$STEPS CKPT=100 \
    MINMEM=${MINMEM:-10} bash $D/trainarm.sh A $T/arms/sources-A.json ) > $RD/armA.out 2>&1 &
tp=$!; lg=$RD/inksmokeA.log
sps_of() { grep -E '^step +[0-9]+ ' $1 | awk '{ if (match($0, /fresh [0-9.]+ samp\/s/)) { split(substr($0, RSTART, RLENGTH), f, " "); print f[2] }
    else { n = split($0, w, " "); for (i = n; i > 1; i--) if (w[i] == "samp/s") { print w[i - 1]; break } } }'; }
sanity() { local f=$O/smokeA.txt
  { echo "arm A desk smoke (card 1, scrolls 1667 0814 MAN5 0009B): $(date '+%F %T'), trainer + $(( ($(date +%s) - t0) / 60 )) min"
    echo "steps logged: $(grep -cE '^step +[0-9]+ ' $lg), last: $(grep -E '^step +[0-9]+ ' $lg | tail -1 | cut -c1-300)"
    echo "samples/s (fresh) per logged line: $(sps_of $lg | tr '\n' ' ')"
    grep -E '^step +[0-9]+ ' $lg | grep -oE 'ink bce [0-9.naif-]+' | awk '{v[NR] = $3} END {if (NR < 6) {print "ink bce: only " NR " logged lines"; exit}
      a = (v[1] + v[2] + v[3]) / 3; b = (v[NR] + v[NR - 1] + v[NR - 2]) / 3; printf "ink bce: first 3 lines %.4f, last 3 %.4f (%s)\n", a, b, b < a ? "falling: OK" : "NOT falling: WARN"}'
    echo "non-finite lines: $(grep -ciE 'non-finite|^step .*( nan| inf)' $lg)"
    echo "peak card-1 use seen: ${PEAK:-?} MiB; host RSS peak of the trainer: ${RSSPK:-?} GB"
  } > $f; cat $f >> $L; cp $lg $RD/inksmokeA.resume.log $O/ 2>/dev/null
  ls $RD/inksmokeA/snap/*.ckpt > /dev/null 2>&1 && cp $(ls -t $RD/inksmokeA/snap/*.ckpt | head -1) $O/; }
stoparm() { touch $RD/inksmokeA.stop; kill $(cat $RD/inksmokeA.pid 2>/dev/null) 2>/dev/null; for i in $(seq 60); do kill -0 $tp 2>/dev/null || break; sleep 2; done
  sanity; fin "FAILED: $*" 1; }
PEAK=0; RSSPK=0; g10=0; g50=0
while kill -0 $tp 2>/dev/null; do
  st=$(awk '/^step +[0-9]+ / {s = $2} END {print s + 0}' $lg 2>/dev/null)
  u=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "${u:-0}" -gt $PEAK ] && PEAK=$u
  tpid=$(cat $RD/inksmokeA.pid 2>/dev/null); [ -n "$tpid" ] && r=$(awk '/^(RssAnon|RssShmem)/ {s += $2} END {printf "%.1f", s / 1048576}' /proc/$tpid/status 2>/dev/null) \
    && awk -v a=${r:-0} -v b=$RSSPK 'BEGIN {exit !(a > b)}' && RSSPK=$r
  if [ "${st:-0}" -ge 10 ] && [ $g10 = 0 ]; then g10=1; sps=$(sps_of $lg | head -1)
    { grep -E '^step +[0-9]+ ' $lg | head -1; grep -E '^memory:' $lg | tail -2; } > $O/steptimeA.txt
    say "step $st at trainer + $(( ($(date +%s) - t0) / 60 )) min: $sps samples/s (gate $MINSPS10), card 1 peak $PEAK MiB, RSS $RSSPK GB"
    awk -v a=${sps:-0} -v b=$MINSPS10 'BEGIN {exit !(a < b)}' && stoparm "below the step-10 gate: $sps samples/s"; fi
  if [ "${st:-0}" -ge 50 ] && [ $g50 = 0 ]; then g50=1; sps=$(sps_of $lg | awk '{s += $1; n++} END {printf "%.3f", n ? s / n : 0}')
    echo "mean samples/s to step $st: $sps" >> $O/steptimeA.txt
    say "step $st at trainer + $(( ($(date +%s) - t0) / 60 )) min: mean $sps samples/s (gate $MINSPS), card 1 peak $PEAK MiB, RSS $RSSPK GB"
    awk -v a=$sps -v b=$MINSPS 'BEGIN {exit !(a < b)}' && stoparm "below the step-50 gate: $sps samples/s at step $st"; fi
  [ $(( $(date +%s) - t0 )) -gt $ARMMAX ] && stoparm "over ARMMAX $ARMMAX s at step $st"
  sleep 20
done
wait $tp; st=$(awk '/^step +[0-9]+ / {s = $2} END {print s + 0}' $lg 2>/dev/null)
sanity
[ "$(grep -ciE 'non-finite|^step .*( nan| inf)' $lg)" = 0 ] || fin "FAILED: non-finite losses ($O/smokeA.txt)" 1
[ "${st:-0}" -ge $STEPS ] || fin "FAILED: the arm stopped at step $st ($lg, $RD/armA.out)" 1
fin "DONE: arm A reached step $st in $(( ($(date +%s) - t0) / 60 )) min; $O/smokeA.txt; card 1 released"

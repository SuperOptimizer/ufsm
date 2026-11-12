#!/usr/bin/env bash
# scripts/univ/u2ink_pod.sh [ARM...]: the U2 ink smoke (scripts/univ/u2ink_smoke.sh) on the shared pod GPU, in units per /workspace/ue/GPU.md:
# training in UNIT-step pieces (--stop-at, the schedule of the full STEPS; resumed from last.ckpt), then each scored checkpoint one crop
# at a time (baseline.sh ONLY=crop). Before every unit: wait while gpu.pause exists, then gpu.shared/inkB for the unit's length.
# Restartable: finished units are skipped (the run's log, the crops' vscores rows). Scores at every 200 steps (EVERY) and the last.
# Env: K (/workspace/univ: binary-independent kit, klt1.lcb, CT cache), IB (/workspace/inkB: the ink kit, runs, the milfix binary),
#      U (IB/ufsm/build/ufsm), STEPS (400), UNIT (100), EVERY (200), GPU (0), WORKERS / SLOTS (3 / 3: the sampler's host memory beside
#      U2), MEMMAX / MEMNEED (160 / 25 GB: a unit starts only while the cgroup's memory + MEMNEED <= MEMMAX); the rest as u2ink_smoke.sh.
set -uo pipefail; D=$(dirname "$(readlink -f "$0")"); ARMS=${*:-A B}
export K=${K:-/workspace/univ}; IB=${IB:-/workspace/inkB}; export IK=$IB/inkgt R=$IB/runs U=${U:-$IB/ufsm/build/ufsm} INIT=${INIT:-$IB/ckpt/u1-9000.ckpt}
export STEPS=${STEPS:-400} EVERY=${EVERY:-200} GPU=${GPU:-0} P=${P:-1024} WORKERS=${WORKERS:-3} SLOTS=${SLOTS:-3}; UNIT=${UNIT:-100}; G=/workspace/ue; mkdir -p $R
say() { echo "== $(date '+%F %T') u2ink_pod: $*"; }
MEMMAX=${MEMMAX:-160}; MEMNEED=${MEMNEED:-25}   # (GB: the pod's cgroup is shared with the resident U2, up to ~120 GB: start no unit that could pass MEMMAX)
cgmem() { echo $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 )); }
unit() {   # one GPU unit under the sharing protocol, with the host-memory guard; its peak RSS (the unit's ufsm processes) to R/units.tsv
  while [ -e $G/gpu.pause ] || [ $(( $(cgmem) + MEMNEED )) -gt $MEMMAX ]; do sleep 30; done
  touch $G/gpu.shared/inkB; local t0=$(date +%s) pk=$R/.peak.$$; echo 0 > $pk
  ( while :; do r=$(ps -eo rss,args | awk '/ufsm train .*u2ink|ufsm predict .*u2ink|ink-vscore/ && !/awk/ {s += $1} END {print int(s / 1048576)}'); [ "$r" -gt "$(cat $pk)" ] && echo $r > $pk; sleep 5; done ) & local mon=$!
  "$@"; local rc=$?; kill $mon 2> /dev/null; rm -f $G/gpu.shared/inkB
  printf '%s\t%s\t%d s\tpeak %s GB RSS\tcgroup %s GB\trc %d\n' "$(date '+%F %T')" "$UNITNAME" $(( $(date +%s) - t0 )) "$(cat $pk)" "$(cgmem)" $rc >> $R/units.tsv; rm -f $pk
  return $rc
}
trap 'rm -f $G/gpu.shared/inkB' EXIT
laststep() { awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $R/u2ink$1.log 2>/dev/null || echo 0; }
for arm in $ARMS; do o=$R/u2ink$arm; mkdir -p $o/snap
  for ((t = UNIT; t <= STEPS; t += UNIT)); do
    [ "$(laststep $arm)" -ge $t ] && continue
    say "arm $arm: train to $t"; UNITNAME="train $arm $t"; unit env STOPAT=$t SCORE=0 bash $D/u2ink_smoke.sh $arm || { say "arm $arm: training unit to $t failed"; exit 1; }
    [ "$(laststep $arm)" -ge $t ] || { say "arm $arm: no step $t in the log"; exit 1; }
    if [ $((t % EVERY)) = 0 ] || [ $t = $STEPS ]; then cp $o/last.ckpt $o/snap/$t.ckpt; fi
  done
done
for arm in $ARMS; do o=$R/u2ink$arm
  for ck in $(ls $o/snap/*.ckpt | sort -V); do st=$(basename $ck .ckpt)
    ch=$(head -c 200000 $ck | head -1 | grep -o '"ink":[0-9]*' | head -1 | cut -d: -f2); [ -n "$ch" ] || { say "no ink channel in $ck"; continue; }
    so=$o/score/$st; tsv=$so/ckpt_${st}_ch$ch/vscores.tsv
    for c in $(dirname $IK)/inkeval/crops/w*_? $IK/evcrops/PHerc*; do n=$(basename $c)
      [ -f $tsv ] && grep -qP "^$n\t" $tsv && continue
      UNITNAME="score $arm $st $n"; unit env D=$IK CROPS=$(dirname $IK)/inkeval/crops CACHE=$K/cache U=$U O=$so ONLY=$n bash $D/../inkgt/baseline.sh ckpt:$ck:$ch $GPU > /dev/null 2>> $so.err
    done
    python3 -I - $tsv $arm $st <<'PY'
import sys, collections
a = collections.defaultdict(list); f1 = collections.defaultdict(list)
for l in open(sys.argv[1]):
    f = l.rstrip("\n").split("\t"); n = f[0]; s = "Paris4" if n.startswith("w") else "0139" if "0139" in n else "1667" if "1667" in n else n
    a[s].append(float(f[5])); f1[s].append(float(f[7]))
t = {"Paris4": 0.727, "0139": 0.651, "1667": 0.561}
print(f"arm {sys.argv[2]} step {sys.argv[3]}: 3-D AUC / best F1 by scan: " + ", ".join(
    f"{s} {sum(v) / len(v):.3f} / {sum(f1[s]) / len(f1[s]):.3f} (teacher AUC {t.get(s, float('nan')):.3f}, {len(v)} crops)" for s, v in sorted(a.items())) + " (CT floor 0.513)")
PY
  done
done | tee -a $R/u2ink_scores.txt

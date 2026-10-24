#!/usr/bin/env bash
# wcrushsched.sh: GPU ${GPU:-1} = the crush arm (wcrush.sh) + its evals (wradsched.sh with the name wcrush). Every 2 min: when
# runs/wcrush/last.ckpt reaches a multiple of EVERY (4000) not yet evaluated, stop the trainer (runs/wcrush.stop), copy it to
# runs/wcrush/snap-STEP.ckpt, evaluate it on PH1 PH2 PH3 C1 C2 HOLD (evalbox2.sh h2, LABELFREE 0) and resume. Summary vs wind96
# (eval/wind96-h2.txt) in fitval-wcrush.txt: ids and the gross phase error (> 1/4 turn) where the label spacing is < 100 um.
# Survives session restarts (setsid nohup); restarts the arm if it died. Stop: touch wcrushsched.stop (the arm keeps running).
P=/vesuvius/ufsm/paris4/wnew; cd $P; EVERY=${EVERY:-4000}; G=${GPU:-1}; B="PH1 PH2 PH3 C1 C2 HOLD"; mkdir -p eval; touch eval/wcrush-done.txt
export U=${U:-/vesuvius/ufsm/paris4/srcc/build/ufsm} GPU=$G
running() { ps -eo pid,args | awk '$2 == "bash" && $3 ~ /wtrain.sh$/ && $4 == "wcrush" {print $1}'; }
trainer() { ps -eo pid,args | awk -v u="$U" '$2 == u && $3 == "train" && /runs\/wcrush / {print $1}'; }
launch() { rm -f runs/wcrush.stop; setsid nohup ./wcrush.sh > /dev/null 2>&1 < /dev/null & }
ckstep() { head -c 4096 runs/wcrush/last.ckpt 2> /dev/null | grep -ao '"step":[0-9]*' | head -1 | cut -d: -f2; }
summ() {   # summ STEP
  local f=eval/wcrush-$1-h2.txt w=eval/wind96-h2.txt line="$(date +%m-%d_%H:%M) wcrush-$1 |"
  for b in $B; do
    local a=$(awk -v b=$b '$4 == b {print $6}' $f | tail -1) a0=$(awk -v b=$b '$4 == b {print $6}' $w | tail -1)
    local g=$(awk -v b=$b '$4 == b' $f | tail -1 | grep -oE '< 100 um [0-9.]+' | awk '{print $4}') g0=$(awk -v b=$b '$4 == b' $w | tail -1 | grep -oE '< 100 um [0-9.]+' | awk '{print $4}')
    line="$line $b ids ${a:-na} (wind96 $a0) gross<100um ${g:-na} ($g0);"
  done
  echo "$line" >> fitval-wcrush.txt
}
[ -x "$U" ] || { echo "no binary $U (build master with crush first)"; exit 1; }
[ -n "$(running)" ] || launch
while [ ! -e wcrushsched.stop ]; do
  s=$(ckstep)
  [ "${s:-0}" -ge ${STEPS:-100000} ] && break
  if [ -n "$s" ] && [ "$s" -gt 0 ] && [ $((s % EVERY)) = 0 ] && ! grep -qx "$s" eval/wcrush-done.txt; then
    touch runs/wcrush.stop; for p in $(trainer); do kill $p; done
    for i in $(seq 60); do [ -z "$(running)" ] && break; sleep 5; done
    s2=$(ckstep); cp runs/wcrush/last.ckpt runs/wcrush/snap-$s2.ckpt
    LABELFREE=0 flock gpu$G.lock ./evalbox2.sh $P/runs/wcrush/snap-$s2.ckpt wcrush-$s2 2 $G $B > eval/wcrush-$s2.log 2>&1
    summ $s2; echo $s >> eval/wcrush-done.txt; [ "$s2" != "$s" ] && echo $s2 >> eval/wcrush-done.txt
    launch
  fi
  [ -z "$(running)" ] && [ ! -e runs/wcrush.stop ] && launch
  sleep 120
done

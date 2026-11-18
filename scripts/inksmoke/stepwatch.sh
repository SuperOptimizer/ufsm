#!/bin/bash
# stepwatch.sh ARM UNTIL: print each new 10-step line of the pod arm (step, samp/s, Mvox/s, wait, derived-cache line), exit at step >= UNTIL,
# on an exit rc in the resume log, or after STALL (1500) s without a new step line
A=${1:-A}; T=${2:-50}; STALL=${STALL:-1500}; S="ssh -o BatchMode=yes -o ConnectTimeout=20 -i /home/forrest/.ssh/id_ed25519 -p 11377 root@213.173.109.173"
seen=0; t0=$(date +%s)
while :; do
  out=$(timeout 60 $S "grep -E '^step +[0-9]+ |derived cache:' /root/ink/runs/inksmoke$A.log 2>/dev/null; grep -c 'exit rc' /root/ink/runs/inksmoke$A.resume.log 2>/dev/null; cat /sys/fs/cgroup/memory.events | tr '\n' ' '" 2>/dev/null)
  n=$(echo "$out" | grep -cE '^step ')
  if [ "$n" -gt "$seen" ]; then echo "$out" | grep -E '^step |derived' | tail -n $(( (n - seen) * 2 )) | sed -E 's/^(step +[0-9]+).* ([0-9.]+) samp\/s ([0-9.]+) Mvox\/s \(wait ([0-9]+%)\).*/\1: \2 samp\/s, \3 Mvox\/s, wait \4/' | cut -c1-200
    echo "  $(date -u +%T) mem: $(echo "$out" | tail -1)"; seen=$n; t0=$(date +%s); fi
  st=$(echo "$out" | grep -E '^step ' | tail -1 | awk '{print $2}'); rc=$(echo "$out" | grep -E '^[0-9]+$' | tail -1)
  [ "${st:-0}" -ge $T ] && { echo "reached step $st"; exit 0; }
  [ "${rc:-0}" -ge 1 ] && { echo "trainer exited (resume log)"; exit 1; }
  [ $(( $(date +%s) - t0 )) -ge $STALL ] && { echo "STALL: no new step line for $STALL s (last step ${st:-none})"; exit 2; }
  sleep 30
done

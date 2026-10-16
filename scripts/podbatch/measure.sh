#!/usr/bin/env bash
# measure.sh SCROLL: 60 s GPU utilisation / memory, cgroup CPU throttling over the same 60 s (and a 10 s window), predict super-shard progress
s=$1; cd /workspace; L=$(ls -t out/$s/predict-*.log | head -1)
prog() { tr '\r' '\n' < $L | grep -E "^super-shard" | tail -1; }
cs() { awk '{print $2}' /sys/fs/cgroup/cpu.stat | tr '\n' ' '; }
p0=$(prog); c0=$(cs); t0=$(date +%s.%N)
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -lms 1000 > /tmp/gpu.$$ & np=$!
sleep 10; c10=$(cs); sleep 50; kill $np; c1=$(cs); p1=$(prog); t1=$(date +%s.%N)
awk -F', ' '{u += $1; if ($2 > m) m = $2; p += $3; n++} END {printf "GPU util mean %.1f%% (%d samples), mem max %d MiB, power mean %.0f W\n", u / n, n, m, p / n}' /tmp/gpu.$$; rm -f /tmp/gpu.$$
read -r u0 _ _ _ np0 nt0 tu0 _ <<< "$c0"; read -r u10 _ _ _ np10 nt10 tu10 _ <<< "$c10"; read -r u1 _ _ _ np1 nt1 tu1 _ <<< "$c1"
dt=$(echo "$t1 - $t0" | bc)
echo "CPU used $(echo "scale=1; ($u1 - $u0) / 1000000 / $dt" | bc) cores of 23.8; throttled $(( nt1 - nt0 ))/$(( np1 - np0 )) periods, $(( (tu1 - tu0) / 1000 )) ms over ${dt%.*} s; first 10 s: $(( nt10 - nt0 ))/$(( np10 - np0 )) periods, $(( (tu10 - tu0) / 1000 )) ms"
echo "progress: [$p0] -> [$p1]"
grep -E "^(anon|shmem|file) " /sys/fs/cgroup/memory.stat | awk '{printf "%s %.1f GB  ", $1, $2 / 1e9} END {print ""}'

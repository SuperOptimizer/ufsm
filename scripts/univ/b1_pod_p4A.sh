#!/usr/bin/env bash
# scripts/univ/b1_pod_p4A.sh SHARDS (pod 2): Paris 4 June winding phase at 4.8 um over the packed shard list (b1_shardpack.py), on the pod's
# GPU, with the pod disk as a staging area: June segments from the open-data bucket (fetchsegs.sh, June only), wind-seed (the desk run of
# 2026-11-11 is byte-identical to the GT seed.jsonl), the segments removed once seeded (the seed keeps its own mesh copies + q.tif), codes
# at level 1 (raster --level 1, ~43 GB), then the phase in PARTS of the box list. At most two finished parts wait on the disk: the desk
# pulls each part (b1_pull_p4A.sh, rsync --remove-source-files) and touches $O.part-N.pulled here.
# Env: K (/workspace/b1), U (a binary with nn_band.o: master >= labelfast), NP (parts, default 8), T (threads).
set -uo pipefail; SH=${1:?shard list}; K=${K:-/workspace/b1}; U=${U:-/workspace/ufsm-master/build/ufsm}; NP=${NP:-8}; T=${T:-12}
D=$(dirname "$(readlink -f "$0")"); SEG=$K/seg/PHercParis4; G=$K/gt/paris4; AX=$K/axis-0.json; SHP=75784,32693,32693; UM=2.4; mkdir -p $G
step() { echo "== $(date +%T) $*"; }
step fetch June; [ -e $SEG/.done ] || [ -s $G/qA/seed.jsonl ] || bash $D/../pod/fetchsegs.sh PHercParis4 $SEG 20260623 | tail -2
step seed June; if [ ! -s $G/qA/seed.jsonl ]; then A=$(ls -d $SEG/20260623*-w*/mesh/*-on-20260411134726-2.4um.tifxyz | grep -v jordi)
  nice -n 5 $U wind-seed $G/qA $A --vol-um $UM --axis $AX > $G/seedA.log 2>&1 || { tail -3 $G/seedA.log; exit 1; }; fi
[ -s $G/qA/seed.jsonl ] && rm -rf $SEG   # the seed holds mesh copies with q.tif
step raster June 4.8; [ -f $G/codesA48/done ] || { nice -n 5 $U raster $G/codesA48 --shape $SHP --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qA/*.tifxyz > $G/rasterA48.log 2>&1 && touch $G/codesA48/done; tail -1 $G/rasterA48.log; }
[ -f $G/codesA48/done ] || exit 1
O=$G/phaseA48; python3 -I - $SH $NP $K/parts <<'PY'
import sys, os
r = sorted(tuple(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()); n = int(sys.argv[2]); os.makedirs(sys.argv[3], exist_ok=True)
b = []; i = 0
while i < len(r):
    j = i
    while j + 1 < len(r) and r[j + 1][:2] == r[i][:2] and r[j + 1][2] == r[j][2] + 1: j += 1
    z, y, x = r[i]; b.append(f"{z*2048},{y*2048},{x*2048},2048,2048,{(j-i+1)*2048}"); i = j + 1
for p in range(n): open(f"{sys.argv[3]}/part-{p}.boxes", "w").write("\n".join(b[p::n]) + "\n")
PY
for p in $(seq 0 $((NP - 1))); do
  [ -e $O.part-$p.done ] && continue
  while [ $(ls $O.part-*.done 2>/dev/null | wc -l) -ge $(( $(ls $O.part-*.pulled 2>/dev/null | wc -l) + 2 )) ]; do sleep 60; done   # <= 2 unpulled parts
  step phase part $p "($(wc -l < $K/parts/part-$p.boxes) boxes)"; s=$(date +%s)
  for b in $(cat $K/parts/part-$p.boxes); do
    nice -n 5 $U phase-raster $G/codesA48 4.8 $O --gs 2 --pair 1 --radius 160 --axis $AX --threads $T --shards 3 --resume 1 --box $b --gpu 0 --gpu-slots 2 >> $O.log 2>&1 || echo "box $b failed"
  done
  echo "$(date +%T) part $p: $(( $(date +%s) - s )) s" >> $O.parts.log; touch $O.part-$p.done
done
step done

#!/bin/bash
# (test harness of unroll.sh REFSTREAM: replays a finished fields dir as two growing predict parts; see DESIGN 2026-11-12 "the ref streamed")
# feeder O FIELDS SPLIT: two fake predict parts of FIELDS (cells [0, SPLIT), [SPLIT, D)) whose 256-cell slabs "finish" every 3 s, alternately
O=$1 F=$2 SP=$3; D=$(jq -r '.shape[0]' $F/fields.json); h=$(jq -r .h $F/fields.json)
for k in 0 1; do p=$O/p$k; mkdir -p $p; a=$([ $k = 0 ] && echo 0 || echo $SP); b=$([ $k = 0 ] && echo $SP || echo $D)
  for pl in $(ls $F); do [ -e $F/$pl/zarr.json ] || continue; mkdir -p $p/$pl/c; jq -c --argjson n $((b - a)) '.shape[0] = $n' $F/$pl/zarr.json > $p/$pl/zarr.json; done
  jq -c --argjson a $a --argjson n $((b - a)) --argjson h $h '.shape[0] = $n | .origin[0] += $a * $h | .axis = .axis[$a * $h : ($a + $n) * $h + 1]' $F/fields.json > $p/fields.partial.json; : > $p/rows.done; done
s0=0; s1=0; n0=$SP; n1=$((D - SP))
while [ $s0 -lt $n0 ] || [ $s1 -lt $n1 ]; do sleep 3
  for k in 1 0; do p=$O/p$k; if [ $k = 0 ]; then s=$s0; n=$n0; off=0; else s=$s1; n=$n1; off=$SP; fi; [ $s -ge $n ] && continue; e=$((s + 256 < n ? s + 256 : n))
    for z in $(seq $((s / 128)) $(( (e - 1) / 128 ))); do for pl in $(ls $F); do [ -d $F/$pl/c/$((z + off / 128)) ] && ln -sfn $F/$pl/c/$((z + off / 128)) $p/$pl/c/$z; done; done
    echo "$s $e" >> $p/rows.done; if [ $k = 0 ]; then s0=$e; else s1=$e; fi; done; done
touch $O/stream.fin

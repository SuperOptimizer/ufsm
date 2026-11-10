#!/usr/bin/env bash
# judge_wraps.sh: the fibre judge against the final snapped wraps (wrap-dump -> fibwrap.py), one 512-slice slab per scroll. Analysis driver.
set -u
U=/home/forrest/ufsm-fib/build/ufsm; T=/home/forrest/ufsm-fib/tools; W=/vesuvius/ufsm/fibers/work; A=/vesuvius/ufsm/fibers/raw; S=/vesuvius/ufsm/v2/wraps_snap
run() {   # run NAME SCROLL AFV z y0 y1 x0 x1 UM EVERY LEVEL FIRST LAST [TR]
  local n=$1 sc=$2 afv=$3 z=$4 y0=$5 y1=$6 x0=$7 x1=$8 um=$9 ev=${10} lv=${11} first=${12} last=${13} tr=${14:-} d=$W/wd_$1_$4.f32
  [ -s $d ] || nice -n 19 $U wrap-dump $S/$sc/wraps_snap $d --z $z,$((z + 512)) --yx $((y0 - 48)),$((y1 + 48)),$((x0 - 48)),$((x1 + 48)) --every $ev > $W/wd_${n}_$z.log 2>&1 || { echo "$n: wrap-dump failed"; return 1; }
  for p in 0 1 2 3; do
    nice -n 19 ~/bin/memcap -m 8G python3 -I $T/fibwrap.py $A/$afv $d --slab $z,512,512 --region $y0,$y1,$x0,$x1 --level $lv --um $um --first $first --last $last ${tr:+--tr $tr} \
      --part $p/4 --out $W/judgewrap_${n}_$z.$p.tsv > $W/judgewrap_${n}_$z.$p.log 2>&1 &
  done; wait; echo "$n $z done: $(tail -1 $W/wd_${n}_$z.log)"
}
run 0175A PHerc0175A PHerc0175A_1mm_v2.afv 6400 3264 7360 2304 6400 8.64 1 0 33 237
run 1447 PHerc1447 PHerc1447_1mm_v2.afv 12160 1472 5568 1824 5920 8.64 1 0 -1000000 1000000
run 0125 PHerc0125 PHerc0125_3mm_v2.afv 10624 1984 6080 2048 6144 9.362 1 0 40 243
run P4 PHercParis4 Paris4_1mm_v2.afv 11328 1024 7168 1024 7168 9.6 ${P4EV:-1} 2 -1000000 179 /vesuvius/ufsm/fibers/meta/p4_2026_to_2023.json

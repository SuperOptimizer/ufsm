#!/usr/bin/env bash
# judge_native.sh: the fibre judge on the natively registered scrolls against our joined fit sets (fit-phase-box -> fibjudge.py).
# One 512-slice slab per scroll, 4096^2 about the axis, 512^3 tiles. Analysis driver (not part of the pipeline).
set -u
U=/home/forrest/ufsm-fib/build/ufsm; T=/home/forrest/ufsm-fib/tools; W=/vesuvius/ufsm/fibers/work; A=/vesuvius/ufsm/fibers/raw
run() {   # run NAME SET AFV z y x UM
  local n=$1 set=$2 afv=$3 z=$4 y=$5 x=$6 um=$7 raw=$W/fp_${1}_$4.raw
  [ -s $raw ] || UFSM_FIT_THREADS=5 nice -n 19 ~/bin/memcap -m 24G $U fit-phase-box $set $z $y $x 512 4096 4096 $raw --level 0 --step 2 > $W/fp_${n}_$z.log 2>&1 || { echo "$n: fit-phase-box failed"; return 1; }
  for p in 0 1 2 3; do
    nice -n 19 python3 -I $T/fibjudge.py $A/$afv --level 0 --um $um --ours-raw $raw $z,$y,$x,512,4096,4096 2 --slab $z,512,512 --region $y,$((y + 4096)),$x,$((x + 4096)) --shape 30000,12000,12000 \
      --part $p/4 --out $W/judgefit_${n}_$z.$p.tsv --tmp $W/tmp > $W/judgefit_${n}_$z.$p.log 2>&1 &
  done; wait; echo "$n $z done"
}
run 0175A /mnt/scratch/pods/PHerc0175A/unroll_v2/joined PHerc0175A_1mm_v2.afv 6400 3264 2304 8.64
run 1447 /mnt/scratch/pods/PHerc1447/run/joined PHerc1447_1mm_v2.afv 12160 1472 1824 8.64
run 0125 /mnt/scratch/pods/PHerc0125/run/joined PHerc0125_3mm_v2.afv 10624 1984 2048 9.362

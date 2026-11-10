#!/usr/bin/env bash
# trace_1299.sh: the streamline sets of the 2026-11-10 fibre judgement (Paris 4 calibration + PHerc1299), then the judge rows.
# Analysis driver; reads the public open-data fibre predictions (cached), CPU only.
set -u
cd /vesuvius/ufsm/fibers/work
U=/home/forrest/ufsm-fib/build/ufsm; C=/vesuvius/ufsm/fibers/raw/predcache
tr() { nice -n 19 ~/bin/memcap -m 24G $U fiber-trace $1 $2 $3.fib --box $4 --scaledown $5 --cache $C --threads 2 > $3.log 2>&1; }
tr https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/fibers/20260411134726-fibers-20260915212757-L1 PHercParis4-20260411134726-las-sd1-7ff0ce6c tr_p4_L4 2800,224,480,192,1472,1472 4 &
tr https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1299/representations/predictions/fibers/20260309130042-fibers-20260915212757-L1 PHerc1299-20260309130042-las-sd1-e7a931d5 tr_1299_L4 2704,0,0,192,1287,1287 4 &
tr https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1299/representations/predictions/fibers/20260309130042-fibers-20260915212757-L1 PHerc1299-20260309130042-las-sd1-e7a931d5 tr_1299_L3 5408,0,0,384,2573,2573 3 &
wait
JP="--um 2.4 --level 2 --slab 11328,512 --region 1024,6656,2048,7680 --d-mm 0.3,0.6,1.2,2.4"; JQ="--um 2.399 --level 2 --slab 10944,512 --d-mm 0.3,0.6,1.2,2.4"
nice -n 19 $U fiber-judge tr_p4_L4.fib --phase /vesuvius/ufsm/gt/paris4-wind-20261019 phaseA/9.6 $JP --out fj_p4L4_gt.tsv > fj_p4L4_gt.log 2>&1 &
nice -n 19 $U fiber-judge tr_p4_L4.fib --wraps /vesuvius/ufsm/v2/wraps_snap/PHercParis4/wraps_snap --first 47 --last 179 $JP --out fj_p4L4_wraps.tsv > fj_p4L4_wraps.log 2>&1 &
for l in L3 L4; do
  nice -n 19 $U fiber-judge tr_1299_$l.fib --wraps /vesuvius/ufsm/v2/wraps_src/PHerc1299/wraps $JQ --out fj_1299${l}_wraps.tsv > fj_1299${l}_wraps.log 2>&1 &
  nice -n 19 $U fiber-judge tr_1299_$l.fib --phase /vesuvius/ufsm/latcal/labels/PHerc1299/phase.zarr 9.596 $JQ --out fj_1299${l}_labels.tsv > fj_1299${l}_labels.log 2>&1 &
done
wait; echo done

#!/usr/bin/env bash
# view1299.sh: the picture evidence of 2026-11-10 (tools/fibview.py): PHerc1299 worst / median / best judge tile and two Paris 4
# reference tiles. Reads the public native CT chunks (cached under /vesuvius/ufsm/fibers/raw/ctcache). Analysis driver, CPU.
cd /vesuvius/ufsm/fibers/work || exit 1
S=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com; V=/home/forrest/ufsm-fib/tools/fibview.py; O=view1299
Q=$S/PHerc1299/volumes/20260309130042-2.399um-0.2m-78keV-masked.zarr/0; WS=/vesuvius/ufsm/latcal/wraps_snap/PHerc1299/wraps_snap
P=$S/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0; WP=/vesuvius/ufsm/v2/wraps_snap/PHercParis4/wraps_snap
rm -f $O/index.txt $O/*.png
nice -n 19 python3 -I $V 1299worst  $Q $WS 0 100000 tr_1299_L3.fib 2.399 10944,3584,2048 $O 2 2
nice -n 19 python3 -I $V 1299median $Q $WS 0 100000 tr_1299_L3.fib 2.399 10944,2560,2560 $O 2 2
nice -n 19 python3 -I $V 1299best   $Q $WS 0 100000 tr_1299_L3.fib 2.399 10944,3072,2560 $O 2 2
nice -n 19 python3 -I $V paris4median $P $WP 47 179 tr_p4_L4.fib 2.4 11328,3072,5632 $O 2 2
nice -n 19 python3 -I $V paris4best   $P $WP 47 179 tr_p4_L4.fib 2.4 11328,5120,6144 $O 2 2

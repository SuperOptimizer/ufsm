#!/usr/bin/env bash
# predict --boxes FILE (many boxes in one process: the CUDA context and the memory pool carry over) == one process per box, every output file
# byte for byte: a sub-cell net with the fine stem, rv 2 and ink (--heads all) plus --fit-zarr per box; three boxes of different sizes, then
# the first box's size again (pooled buffers reused: device blocks zeroed), the second box with the GPU encoder at q 8.
# Usage: test_predict_boxes.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
"$U" train fx/sources-ink.json --out s --task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 \
     --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3 \
     --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --fstem 16 > s.log 2>&1 || fail "train ($D/s.log)"
P=(--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 2 --threads 2 --heads all)
BOX=("0,0,0,256,256,256" "32,0,64,128,192,128" "0,64,0,192,128,256" "8,8,16,248,240,232")
EXTRA=("" "--q 8" "" "")
: > boxes.txt
for i in 0 1 2 3; do
  echo "${BOX[$i]} b/o$i --fit-zarr b/o$i/fz ${EXTRA[$i]}" >> boxes.txt
  "$U" predict s/last.ckpt fx ct p/o$i --box ${BOX[$i]} --fit-zarr p/o$i/fz ${EXTRA[$i]} "${P[@]}" --q 0 > p$i.log 2>&1 || fail "predict box $i ($D/p$i.log)"
done
"$U" predict s/last.ckpt fx ct - --boxes boxes.txt "${P[@]}" --q 0 > b.log 2>&1 || fail "predict --boxes ($D/b.log)"
grep -q "predict --boxes: 4 boxes in" b.log || fail "the --boxes summary line ($D/b.log)"
for i in 0 1 2 3; do
  n=$(find p/o$i -type f | wc -l); [ "$n" -gt 20 ] || fail "box $i: only $n files"
  diff -r p/o$i b/o$i > d$i.txt || fail "box $i: --boxes differs from its own process ($(wc -l < d$i.txt) files, $D/d$i.txt)"
  echo "  box $i (${BOX[$i]} ${EXTRA[$i]}): $n files identical"
done
echo PASS

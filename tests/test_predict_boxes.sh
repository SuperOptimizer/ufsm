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
"$U" train fx/sources-ink.json --out s --task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-train-ok 1 --val-every 1000 \
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
# --tile-list (2026-11-20): one run over box 0 computing only the listed 128-tiles (2 of 8 shards): their level-0 shard files equal the
# whole-box run's, nothing else of level 0 is written
printf '0 0 0\n1 1 0\n' > tl.txt
"$U" predict s/last.ckpt fx ct t --box ${BOX[0]} --tile-list tl.txt --tile-list-size 128 "${P[@]}" --q 0 > t.log 2>&1 || fail "predict --tile-list ($D/t.log)"
grep -q "tile-list tl.txt: 2 of 2 tiles in the box, 6 shards outside them skipped" t.log || fail "the --tile-list line ($D/t.log)"
nt=0; while IFS= read -r f; do
  cmp -s "t/$f" "p/o0/$f" || fail "--tile-list: $f differs from the whole-box run"; nt=$((nt + 1))
done < <(cd t && find . -path '*/c/*' -type f | grep -E '/(0|9\.6|9\.600)/c/' | sed 's#^\./##')
nw=$(cd p/o0 && find . -path '*/c/*' -type f | grep -cE '/(0|9\.6|9\.600)/c/')
[ "$nt" -gt 0 ] && [ "$nt" -lt "$nw" ] || fail "--tile-list: $nt level-0 shard files (whole box: $nw)"
echo "  --tile-list: $nt level-0 shard files identical (whole box $nw)"
echo PASS

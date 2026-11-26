#!/usr/bin/env bash
# predict --seamless 2 --tile-list (perf 11-25) == the whole-box seamless run on the listed tiles, byte for byte: window 288 halo 16 (tile
# 256, shift 128, pass shards 128), output shards 128, 128-voxel tiles (tile = shift: pass b lost them before --tile-list-origin), listed
# tiles at the volume corner, inside, and touching each other; the blend (ink_blend_run, shard presence maps + workers) on both.
# Usage: test_seamless_tilelist.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 512 2 1 > fx.log 2>&1
"$U" train fx/sources-ink.json --out s --task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-train-ok 1 --val-every 1000 \
     --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3 \
     --lat 4 --lat-guide 3 --ink-head 1 --fstem 16 > s.log 2>&1 || fail "train ($D/s.log)"
P=(--um 9.6 --level 0 --window 288 --halo 16 --shard 128 --levels 1 --threads 4 --heads ink --seamless 2 --q 0)
"$U" predict s/last.ckpt fx ct full --box 0,0,0,512,512,512 "${P[@]}" > full.log 2>&1 || fail "predict whole box ($D/full.log)"
printf '0 0 0\n1 2 1\n1 2 2\n3 1 2\n' > tl.txt
"$U" predict s/last.ckpt fx ct tl --box 0,0,0,512,512,512 --tile-list tl.txt --tile-list-size 128 "${P[@]}" > tl.log 2>&1 || fail "predict --tile-list ($D/tl.log)"
grep -q "ink-blend: .* workers" tl.log || fail "no worker line from the blend ($D/tl.log)"
n=0
while read -r a b c; do
  for f in $(cd full && find . -path "*/c/$a/$b/$c" -type f); do
    [ -f "tl/$f" ] || fail "--tile-list: $f missing (tile $a $b $c)"
    cmp -s "tl/$f" "full/$f" || fail "--tile-list: $f differs from the whole-box seamless run (tile $a $b $c)"; n=$((n + 1))
  done
done < tl.txt
[ "$n" -ge 4 ] || fail "only $n listed shard files compared"
nf=$(cd full && find . -path '*/c/*' -type f | wc -l); nt=$(cd tl && find . -path '*/c/*' -type f | wc -l)
[ "$nt" -lt "$nf" ] || fail "--tile-list wrote $nt shard files, the whole box $nf"
echo "  seamless --tile-list: $n listed shard files identical to the whole-box run ($nt of $nf written)"
echo ok

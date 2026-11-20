#!/usr/bin/env bash
# predict --channels A,B,C (one trunk forward per window, several logit channels of a voxel net, 2026-11-20) == one --channel N run per
# channel, every file byte for byte (zarr.json with the run record included): window = shard (one window per shard), a window of 2^3
# shards (the super-shard path) and windows smaller than a shard (several windows placed into each shard). Plus the refusals.
# Usage: test_predict_channels.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
"$U" train fx/sources-ink.json --out s --task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-train-ok 1 --val-every 1000 \
     --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3 --ink-head 1 > s.log 2>&1 \
     || fail "train ($D/s.log)"
B=(--um 9.6 --level 0 --box 0,0,0,256,256,256 --shard 128 --levels 2 --threads 2 --q 0 --axis fx/axis.json)
CH=(2 0 3)
for g in "160 16" "288 16" "96 16"; do
  set -- $g; W=$1; H=$2; R=r$W
  for c in "${CH[@]}"; do
    "$U" predict s/last.ckpt fx ct $R/one$c "${B[@]}" --window $W --halo $H --channel $c > $R.one$c.log 2>&1 || fail "predict --channel $c ($D/$R.one$c.log)"
  done
  "$U" predict s/last.ckpt fx ct $R/m2 "${B[@]}" --window $W --halo $H --channels 2,0,3 --channel-outs $R/m0,$R/m3 > $R.m.log 2>&1 || fail "predict --channels ($D/$R.m.log)"
  grep -q "predict: --channels: 2 more channel stores" $R.m.log || fail "the --channels line ($D/$R.m.log)"
  for c in "${CH[@]}"; do
    n=$(find $R/one$c -type f | wc -l); [ "$n" -gt 4 ] || fail "window $W channel $c: only $n files"
    diff -r $R/one$c $R/m$c > $R.d$c.txt || fail "window $W channel $c: --channels differs from --channel $c ($D/$R.d$c.txt)"
  done
  f0=$(cd $R/one0 && find . -path "*/c/*" -type f | sort | head -1); cmp -s $R/one0/$f0 $R/one2/$f0 && fail "window $W: channels 0 and 2 identical (the test would not see a channel mix-up)"
  echo "  window $W: channels ${CH[*]}: $n files each identical"
done
"$U" predict s/last.ckpt fx ct x "${B[@]}" --channels 0,1 --tta 2 > x1.log 2>&1 && fail "--channels with --tta accepted"
grep -q "not with --tta" x1.log || fail "the --tta refusal ($D/x1.log)"
"$U" predict s/last.ckpt fx ct x "${B[@]}" --channels 0,99 > x2.log 2>&1 && fail "--channels 99 accepted"
grep -q "outside the net's" x2.log || fail "the channel range refusal ($D/x2.log)"
"$U" predict s/last.ckpt fx ct x "${B[@]}" --channels 0,1 --channel-outs a,b > x3.log 2>&1 && fail "--channel-outs with too many dirs accepted"
echo PASS

#!/usr/bin/env bash
# ufsm gnw-stats: 8 windows along x (window 16, halo 0, --ct-div 1) over a 16 x 16 x 128 CT whose occupancy rises with x; the windows'
# GroupNorm statistics sit further from the median the emptier the window is -> D falls with occupancy (rank -0.87: ties among the
# typical ones), the empty bin holds the atypical windows; one GT box per window, arm B = A + 0.05: paired mean +0.05, ids follow occupancy.
set -u; U=${1:-build/ufsm}; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; f=0
le32() { printf "\\x$(printf %02x $(($1 & 255)))\\x$(printf %02x $((($1 >> 8) & 255)))\\x$(printf %02x $((($1 >> 16) & 255)))\\x$(printf %02x $((($1 >> 24) & 255)))"; }
# float bits: 0, 10, 20, 30, 40
F=(0 1092616192 1101004800 1106247680 1109393408)
for k in 0 1 2 3 4 5 6 7; do s=$(( k < 4 ? 4 - k : 0 )); { le32 0; le32 0; le32 $((16 * k)); le32 2; le32 ${F[$s]}; le32 ${F[$s]}; } >> $T/gnw.bin; done
# CT: x-slab of window k filled (value 100) in its first 2k of 16 columns: occupancy k / 8
for k in 0 1 2 3 4 5 6 7; do for c in $(seq 0 15); do [ $c -lt $((2 * k)) ] && printf '\x64' || printf '\x00'; done; done > $T/row
for i in $(seq 256); do cat $T/row; done > $T/ct.raw
[ "$(stat -c %s $T/ct.raw)" = 32768 ] || { echo "FAIL fixture size"; exit 1; }
mkdir -p $T/a $T/b; : > $T/boxes.tsv
for k in 0 1 2 3 4 5 6 7; do
  printf 'b%d\ts\t0\t0\t%d\t16\t16\t16\t1\n' $k $((16 * k)) >> $T/boxes.tsv
  printf 'name\tvoxels\tids\nb%d\t4096\t0.%02d\n' $k $((20 + 10 * k)) > $T/a/b$k.tsv; printf 'name\tvoxels\tids\nb%d\t4096\t0.%02d\n' $k $((25 + 10 * k)) > $T/b/b$k.tsv
done
o=$($U gnw-stats $T/gnw.bin --ct-raw $T/ct.raw --ct-shape 16,16,128 --ct-div 1 --window 16 --halo 0 --boxes $T/boxes.tsv --arm A=$T/a --arm B=$T/b 2>&1); echo "$o"
echo "$o" | grep -q '8 windows, 2 statistics' || { echo "FAIL window count"; f=1; }
echo "$o" | grep -qE 'corr\(D, occ>20\) -0\.8[0-9]* \(rank -0\.8' || { echo "FAIL D vs occupancy"; f=1; }
echo "$o" | grep -qE 'boxes: 8 of 8 scored in every arm \(2 arms\), median window coverage 1\.00' || { echo "FAIL boxes"; f=1; }
echo "$o" | grep -qE 'B-A: mean \+0\.0500, better / worse 8 / 0' || { echo "FAIL paired difference"; f=1; }
echo "$o" | grep -qE 'A +ids mean 0\.5500: corr\(ids, D\) -[0-9.]+ \(rank -[0-9.]+\), corr\(ids, occ\) \+1\.000' || { echo "FAIL ids vs occupancy"; f=1; }
[ $f = 0 ] && echo "test_gnwstats: ok"; exit $f

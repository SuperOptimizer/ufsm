#!/bin/bash
# wb_boxes.sh SCROLL [BOX...]: the wbench boxes of a scroll (paris4 covers paris4j) as ufsm latvol --boxes lines, each with the predict
# halo below and halo + one window stride above (window 640, halo 64: the windows of a box reach past its end)
H=$(cd "$(dirname "$0")/../wbench" && pwd); s=$1; shift; bl=" $* "
awk -F'\t' -v s=$s -v bl="$bl" '$1 !~ /^#/ && ($2 == s || (s == "paris4" && $2 == "paris4j")) && (bl == "  " || index(bl, " " $1 " ")) { print $3 - 64, $4 - 64, $5 - 64, $6 + 64 + 576, $7 + 64 + 576, $8 + 64 + 576 }' $H/boxes.tsv

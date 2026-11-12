#!/bin/bash
# wb_boxes.sh SCROLL [BOX...]: the wbench boxes of a scroll (paris4 covers paris4j) as ufsm latvol --boxes lines: the region predict
# reads for the box with wbench's windows (window 640, halo 64, --grid-origin 0: tiles on the 512 grid, each read with its halo)
H=$(cd "$(dirname "$0")/../wbench" && pwd); s=$1; shift; bl=" $* "
awk -F'\t' -v s=$s -v bl="$bl" 'function lo(a) { return int(a / 512) * 512 - 64 } function hi(a, n) { return int((a + n + 511) / 512) * 512 + 64 }
  $1 !~ /^#/ && ($2 == s || (s == "paris4" && $2 == "paris4j")) && (bl == "  " || index(bl, " " $1 " ")) { print lo($3), lo($4), lo($5), hi($3, $6) - lo($3), hi($4, $7) - lo($4), hi($5, $8) - lo($5) }' $H/boxes.tsv

#!/usr/bin/env bash
# gtboxes.sh FIELDS MODEL BOXES.tsv CODES_ROOT CODES_KEY OUTDIR [U]
# Score a winding model (a count-field ref-cf, a joined set, a cf-demod model) against GT winding codes on held-out boxes, CPU only
# (fit --model --codes --cpu on mkcrop.sh symlink crops of the run's fields; ~1 min per 512^3 box at 9.6 um). BOXES.tsv: the wbench
# format (name scroll z y x nz ny nx level ...; box in fields-native voxels, i.e. the CT level the fields were predicted at); the box is
# made relative to the fields' origin and divided by the fields' h. Writes OUTDIR/<name>.log and OUTDIR/summary.tsv (name, labelled
# voxels, ids right, |err| < 0.08, |err| median) and prints the voxel-weighted means. Needs a binary with 3a8f25e (fit --model --codes on
# a crop); the 11-12 unroll binary 29115c1 scores 0 voxels. (audit ugeom #2 / #5: ref-cf vs joined on 1667's 70 C boxes and P4's boxes)
set -euo pipefail
F=$(readlink -f "$1") M=$2 BX=$3 CR=$4 CK=$5 OUT=$6 U=${7:-${U:-ufsm}}
H=$(cd "$(dirname "$0")" && pwd); MK=${MKCROP:-$H/../../wbench/mkcrop.sh}
mkdir -p "$OUT/crops"
read -r oz oy ox h < <(jq -r '"\(.origin[0]) \(.origin[1]) \(.origin[2]) \(.h)"' "$F/fields.json")
grep -v '^#' "$BX" | while IFS=$'\t' read -r nm sc z y x nz ny nx rest; do
  [ -n "$nm" ] || continue
  c=$OUT/crops/$nm; [ -d "$c" ] || bash "$MK" "$F" "$c" $(( (z - oz) / h )) $(( (y - oy) / h )) $(( (x - ox) / h )) $(( nz / h )) $(( ny / h )) $(( nx / h )) > /dev/null
  grep -qs 'fit score' "$OUT/$nm.log" || nice -n 10 "$U" fit "$c" --model "$M" --codes "$CR" "$CK" --cpu --score-detail 1 > "$OUT/$nm.log" 2>&1 || true
done
{ printf 'name\tvoxels\tids\terr08\terr_med\n'
  for f in "$OUT"/*.log; do sed -nE "s/^fit score: ([0-9]+) labelled voxels, gauge [-0-9.]+, winding ids right ([0-9.]+), \|error\| < 0.08 turn ([0-9.]+), \|error\| median ([0-9.]+).*/$(basename "$f" .log)\t\1\t\2\t\3\t\4/p" "$f"; done
} > "$OUT/summary.tsv"
awk -F'\t' 'NR > 1 { n += $2; a += $2 * $3; b += $2 * $4; c += $2 * $5; k++ } END { if (n) printf "gtboxes: %d boxes, %.0f labelled voxels: ids right %.4f, |err| < 0.08 %.4f, |err| median (mean) %.4f\n", k, n, a / n, b / n, c / n; else print "gtboxes: nothing scored" }' "$OUT/summary.tsv"

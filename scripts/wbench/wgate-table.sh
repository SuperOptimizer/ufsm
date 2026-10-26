#!/usr/bin/env bash
# wgate-table.sh TAG [REF=wind96]: the wgate decision table (README.md "wgate") from $WB/gate/TAG/{rows,slabs,radius}.tsv vs the
# baseline's $WB/res/REF{,-slabs,-radius}.tsv. Paired per-box differences with stratified bootstrap 95 % CIs (ufsm wbench-stats
# --tsv), the MDE, the slab sites, the radius profile and the pre-registered PASS rule. ROWS= / SLABS= / RADIUS= read other files
# (e.g. an old wbench run: ROWS=$WB/res/s1p-8000.tsv). Env: SITE, STATS (binary with wbench-stats --tsv; default $WB/bin/ufsm-gate).
set -u
H=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE:-$([ -d $HOME/ufsm-data ] && echo laptop || echo desk)}; GPU=0; . $H/site-$SITE.env
TAG=$1; REF=${2:-wind96}; O=$WB/gate/$TAG; ST=${STATS:-$WB/bin/ufsm-gate}
ROWS=${ROWS:-$O/rows.tsv}; SLABS=${SLABS:-$O/slabs.tsv}; RADIUS=${RADIUS:-$O/radius.tsv}
RR=$WB/res/$REF.tsv; RS=$WB/res/$REF-slabs.tsv; RRAD=$WB/res/$REF-radius.tsv; ORC=$WB/res/slaborc-net-slabs.tsv
[ -s $ROWS ] || { echo "no rows $ROWS"; exit 2; }
# the baseline's ceilings decide the clean set for both sides (rows without a ceil get REF's)
T=$(mktemp -d); trap 'rm -rf $T' EXIT
awk -F'\t' 'NR == FNR { if (FNR > 1) c[$2] = $22; next } FNR == 1 { for (i = 1; i <= NF; i++) if ($i == "ceil") k = i; print; next }
  { if (k && ($k == "na" || $k == "") && ($2 in c)) $k = c[$2]; print }' OFS='\t' $RR $ROWS > $T/a.tsv
st() { $ST wbench-stats $T/a.tsv $RR --metric $1 --scroll $2 --tsv 1 --n 10000 | tail -n +2; }
{ st p05x,ids,p1x paris4,paris4j; st p05x,ids paris4j; st p05x,ids 1667; st cnt_off,cnt_bias paris4,paris4j; } > $T/s.tsv
nbox=$(awk -F'\t' 'NR > 1' $ROWS | wc -l)
echo "wgate $TAG vs $REF ($nbox box rows; $(cat $O/ckpt 2> /dev/null || echo "rows $ROWS"))"
printf '%-24s %-8s %3s %7s %7s %8s  %-19s %6s %6s\n' set metric n $REF $TAG diff "95% CI" "P(>)" MDE
awk -F'\t' '{ s = $2 == "paris4,paris4j" ? "primary paris4+paris4j" : $2 == "paris4j" ? "  (paris4j only)" : $2 == "1667" ? "guard 1667" : $2
  if ($1 ~ /^cnt/) s = "diag (no rule)"
  printf "%-24s %-8s %3s %7s %7s %8s  [%7s, %7s] %6s %6s\n", s, $1, $3, $7, $4, $8, $9, $10, $11, $12 }' $T/s.tsv
v() { awk -F'\t' -v m=$1 -v s=$2 -v c=$3 '$1 == m && $2 == s {print $c}' $T/s.tsv; }
# ---- slab sites: joined ids (the direct scale-failure metric), join share, the label-phase oracle ceiling ----
echo
printf '%-10s %8s %8s %8s %8s %8s %10s\n' site "$REF" "$TAG" oracle join_ref join_tag gap_closed
[ -s $SLABS ] && awk -F'\t' 'FNR == 1 { f++; next } f == 1 { r[$2] = $10; rj[$2] = $9 } f == 2 { o[$2] = $10 } f == 3 && ($2 in r) {
    g = (o[$2] != "" && o[$2] - r[$2] > 0) ? sprintf("%+.3f", ($10 - r[$2]) / (o[$2] - r[$2])) : "na"
    printf "%-10s %8.4f %8s %8s %8.4f %8s %10s\n", $2, r[$2], $10, o[$2], rj[$2], $9, g }' $RS $ORC $SLABS
[ -s $SLABS ] || SLABS=/dev/null
read -r sn sa sb <<< "$(awk -F'\t' 'FNR == 1 { f++; next } f == 1 { r[$2] = $10 } f == 2 && ($2 in r) && $10 != "na" { n++; a += r[$2]; b += $10 }
  END { if (n) printf "%d %.4f %.4f\n", n, a / n, b / n; else print "0 na na" }' $RS $SLABS)"
echo "slab joined mean over $sn sites: $REF $sa  $TAG $sb"
# ---- radius profile: share of slab voxels with the right winding (zero integer error), voxel weighted, by radius band ----
if [ -s $RADIUS ] && [ -s $RRAD ]; then
  echo; echo "radius profile (share with zero winding error, slab fits, voxel weighted)"
  for f in $RRAD $RADIUS; do awk -F'\t' -v t=$(basename $f) 'NR > 1 { r = $4; b = r < 5 ? "0-5mm" : r < 10 ? "5-10mm" : r < 20 ? "10-20mm" : "20+mm"; w[b] += $6; z[b] += $6 * $9 }
    END { printf "  %-26s", t; split("0-5mm 5-10mm 10-20mm 20+mm", B, " "); for (i = 1; i <= 4; i++) printf " %s %.3f", B[i], w[B[i]] ? z[B[i]] / w[B[i]] : 0; printf "\n" }' $f; done
fi
# ---- the pre-registered rule (README.md "wgate") ----
echo
pd=$(v p05x paris4,paris4j 8); plo=$(v p05x paris4,paris4j 9); phi=$(v p05x paris4,paris4j 10); id=$(v ids paris4,paris4j 8); md=$(v p05x paris4,paris4j 12)
ghi=$(v p05x 1667 10); gihi=$(v ids 1667 10); np=$(v p05x paris4,paris4j 3); ng=$(v p05x 1667 3)
r1=$(awk -v lo=$plo -v d=$id 'BEGIN { print (lo + 0 > 0 && d + 0 >= 0) ? "PASS" : "fail" }')
r2=$(awk -v a=$sa -v b=$sb -v n=$sn 'BEGIN { print (n >= 4 && b + 0 > a + 0) ? "PASS" : (n < 4 ? "incomplete" : "fail") }')
r3=$(awk -v a=$ghi -v b=$gihi 'BEGIN { print (a + 0 >= 0 && b + 0 >= 0) ? "PASS" : "fail" }')
echo "R1 primary ($np boxes): p05x diff CI low $plo > 0 and ids diff $id >= 0 -> $r1"
echo "R2 slabs: joined mean $sb > $REF $sa (all 4 sites) -> $r2"
echo "R3 guard 1667 ($ng boxes): not worse beyond the CI (p05x CI high $ghi >= 0, ids CI high $gihi >= 0) -> $r3"
if [ $r1 = PASS ] && [ $r2 = PASS ] && [ $r3 = PASS ]; then vd="PASS: promote (next: p4unwrap milestone)"
elif [ $r3 = fail ] || awk -v h=$phi 'BEGIN { exit !(h + 0 < 0) }'; then vd="FAIL: worse than $REF"
elif awk -v d=$pd 'BEGIN { exit !(d + 0 > 0) }'; then vd="INCONCLUSIVE: p05x +$pd is inside the noise (MDE $md); no promotion"
else vd="FAIL: no gain over $REF"; fi
echo "VERDICT $TAG: $vd"

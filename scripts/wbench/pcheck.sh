#!/bin/bash
# pcheck.sh CKPT TAG: the fast network-only phase check (no fitter; a few minutes on one GPU). For each box (default PH1 PH2 PH3 HOLD +
# the 6 most packed clean 1667 boxes): predict fit fields (wbench's production predict), the label phase as an oracle field set,
# fit-compare -> the turn counts on 0.5 mm radial segments (cnt_*: by the network phase, dcnt_*: by its density; bias = median turns
# error, off = share >= 0.5 turn off, offp = the same on packed < 100 um segments), gross (> 1/4 turn) phase errors overall and on
# < 100 um spacing (gross100), the median phase error (pmed) and confidence. Rows -> $WB/res/pcheck-TAG.tsv, then the means per group.
# Uses wbench.sh's scroll / predict functions (same sites, data and pinned binary). Env: SITE (laptop | desk), GPU, U (binary),
# BOXES (names from boxes.tsv), KEEP=1 keeps the fields ($WB/pcheck/TAG/BOX), FIELDS=TAG0 (no predict: wbench TAG0's fields; no GPU), NOLOCK=1 (the caller holds the GPU lock), LOCK.
# Needs a free GPU (predict has no CPU path); on the desk use the GPU nobody trains on, or the laptop via gpu.lock.
set -u
H=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE:-$([ -d $HOME/ufsm-data ] && echo laptop || echo desk)}; GPU=${GPU:-0}
. $H/site-$SITE.env
CK=$(readlink -f "$1"); TAG=$2
U=${U:-$WB/bin/ufsm}; [ -x $U ] || { echo "no binary $U"; exit 2; }
BOXES=${BOXES:-PH1 PH2 PH3 HOLD C4 C19 C14 C3 C2 C24}; BT=${BT:-$H/boxes.tsv}; KEEP=${KEEP:-0}
eval "$(sed -n '/^scroll() {/,/^# ---- boxes ----/p' $H/wbench.sh)"   # scroll, predict, g_cnt, num (wbench.sh is the reference)
O=$WB/pcheck/$TAG; RES=$WB/res/pcheck-$TAG.tsv; mkdir -p $O $WB/res $CACHE
[ -s $RES ] || printf 'tag\tbox\tscroll\tcnt_bias\tcnt_off\tcnt_offp\tdcnt_bias\tdcnt_off\tdcnt_offp\tgross\tgross100\tpmed\tconf\tpred_s\n' > $RES
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $O/run.log; }
# GPU lock per box around predict only (as wbench.sh; released in between so other jobs interleave); NOLOCK=1: the caller holds it
LK=; [ -z "${NOLOCK:-}" ] && [ -n "${LOCK:-}" ] && LK=$LOCK; HL=0
lk() { [ -n "$LK" ] && [ $HL = 0 ] || return 0; exec 9> $LK; flock -n 9 || { log "waiting for $LK"; flock 9; }; HL=1; }
ulk() { [ $HL = 1 ] || return 0; flock -u 9; exec 9>&-; HL=0; sleep 3; }
log "pcheck $TAG ckpt $CK boxes $BOXES site $SITE gpu $GPU"
for B in $BOXES; do
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"; [ -n "${nx:-}" ] || { log "no box $B"; continue; }
  grep -qP "^$TAG\t$B\t" $RES && { log "$B done"; continue; }
  scroll $sc || { log "$B: unknown scroll $sc"; continue; }
  d=$O/$B; f=$d/fields; [ -n "${FIELDS:-}" ] && f=$WB/fields/$FIELDS/$B; mkdir -p $d; t0=$(date +%s)
  [ -e $f/fields.json ] || { lk; predict $f $z $y $x $nz $ny $nx $d/pred.log || { log "$B predict failed (see $d/pred.log)"; ulk; continue; }; ulk; }
  t1=$(date +%s)
  RUN $U fields-oracle $f $PR $PK $d/orc --fill none --phase-scale $PF > $d/orc.log 2>&1 && RUN $U fit-compare $d/orc $f > $d/cmp.log 2>&1 || { log "$B compare failed"; continue; }
  rm -rf $d/orc
  pe=$(num $(grep -oE 'phase \|error\| \(turns\): median [0-9.]+' $d/cmp.log | awk '{print $NF}')); gr=$(num $(grep -oE '> 0.25 [0-9.]+' $d/cmp.log | head -1 | awk '{print $3}'))
  g1=$(num $(grep 'gross phase errors by reference spacing' $d/cmp.log | grep -oE '< 100 um [0-9.]+' | awk '{print $4}'))
  cf=$(num $(grep -oE 'phase confidence: median [0-9.]+' $d/cmp.log | awk '{print $NF}'))
  read -r cnb cno cnp <<< "$(g_cnt $d/cmp.log phase)"; read -r dnb dno dnp <<< "$(g_cnt $d/cmp.log density)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $TAG $B $sc $cnb $cno $cnp $dnb $dno $dnp $gr $g1 $pe $cf $((t1 - t0)) >> $RES
  log "$B cnt_bias $cnb cnt_off $cno cnt_offp $cnp dcnt_bias $dnb gross $gr gross100 $g1 pmed $pe pred $((t1 - t0))s"
  [ "$KEEP" = 1 ] || [ -n "${FIELDS:-}" ] || rm -rf $f
done
# means per group (Paris 4 in-sample PH*, HOLD, 1667)
awk -F'\t' -v t=$TAG 'NR > 1 && $1 == t { g = $2 ~ /^PH/ ? "PH1-3" : $2 == "HOLD" ? "HOLD" : $3; n[g]++; for (k = 4; k <= 12; k++) if ($k != "na") { s[g, k] += $k; c[g, k]++ } }
  END { printf "%-6s %3s %8s %7s %8s %9s %8s %9s %6s %8s %6s\n", "group", "n", "cnt_bias", "cnt_off", "cnt_offp", "dcnt_bias", "dcnt_off", "dcnt_offp", "gross", "gross100", "pmed";
        for (g in n) { printf "%-6s %3d", g, n[g]; for (k = 4; k <= 12; k++) printf " %8s", c[g, k] ? sprintf("%.3f", s[g, k] / c[g, k]) : "na"; printf "\n" } }' $RES | tee -a $O/run.log
log "pcheck $TAG finished"

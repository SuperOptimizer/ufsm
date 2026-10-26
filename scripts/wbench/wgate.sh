#!/usr/bin/env bash
# wgate.sh CKPT TAG: the winding-model gate (README.md "wgate"): one candidate checkpoint vs the cached wind96 baseline -> one
# decision table with the pre-registered PASS rule. Boxes: the 44 core boxes (clean paris4 + paris4j = primary, 12 1667 core boxes =
# guard) and the 4 Paris 4 slab sites (joined ids, join share, radius profile). The baseline is never re-run: its rows
# ($WB/res/$REF.tsv, -slabs, -radius; fits are deterministic and identical across binaries) and its label ceilings are reused.
# Pipeline (one GPU block, gpuq priority eval): the GPU lane only predicts (the 4 sites first, then the boxes); FL box lanes fit each
# predicted box (the box fit + 2 shifted sub-domain fits, concurrently) while the GPU predicts the next ones; one slab lane fits the
# sites (2 slabs, fit-join, joined score) as their fields land. The phase / turn-count columns (fields-oracle + fit-compare, CPU only, ~25 s a box) run in the box lanes off the GPU (DIAG=0 skips them); no ceiling fits.
# Output: $WB/gate/TAG/{rows.tsv (wbench.sh columns), slabs.tsv, radius.tsv, table.txt, run.log}; fields in $WB/fields/TAG (reused
# by a rerun; KEEP=0 deletes them at the end). A rerun skips boxes / sites that already have rows.
# Env: SITE (laptop), GPU (0), U (fitter binary; default the pin $WB/bin/ufsm), UP (predict binary; default $WB/bin/ufsm-gate),
#   REF (wind96), BOXES (core | names), SITES ("4624 5392 5904 6672"; "" = no slabs), REP (2), FL (box lanes, 3), DIAG (1),
#   KEEP (1), NOLOCK=1 (the caller holds the GPU), PRIO (eval), TTA (predict --tta K, 1 = off), SEAM (1: average 2 window-grid
#   offsets 0 / 8 vx: volcomp seam TTA, 2x predict), SPRIO (slab fits after the block, bg).
set -u
H=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE:-$([ -d $HOME/ufsm-data ] && echo laptop || echo desk)}; GPU=${GPU:-0}
. $H/site-$SITE.env
[ $# -ge 2 ] || { sed -n '2,15p' $0; exit 2; }
CK=$(readlink -f "$1"); TAG=$2
U=${U:-$WB/bin/ufsm}; UP=${UP:-$WB/bin/ufsm-gate}; [ -x $UP ] || UP=$U; [ -x $U ] || { echo "no binary $U"; exit 2; }
TTA=${TTA:-1}; SEAM=${SEAM:-0}; [ "$TTA" -gt 1 ] && export PX="--tta $TTA"   # (wbench.sh's predict appends PX)
REF=${REF:-wind96}; BOXES=${BOXES:-core}; SITES=${SITES-4624 5392 5904 6672}; REP=${REP:-2}; FL=${FL:-3}; DIAG=${DIAG:-1}; KEEP=${KEEP:-1}
BT=${BT:-$H/boxes.tsv}; FIT=""; FIELDS=""
GF="--gpu $GPU"; BIN=$(sha1sum $U | cut -c1-8); FCFG=$(echo "$BIN --dir 1 --uw-repair 8 " | sha1sum | cut -c1-8)
O=$WB/gate/$TAG; FD=$WB/fields/$TAG; mkdir -p $O/ready $O/claim $FD $CACHE
RES=$O/rows.tsv; RS=$O/slabs.tsv; RR=$O/radius.tsv; RREF=$WB/res/$REF.tsv
[ -s $RREF ] || { echo "no baseline rows $RREF"; exit 2; }
eval "$(sed -n '/^scroll() {/,/^# ---- boxes ----/p' $H/wbench.sh)"   # scroll, predict, fitsc, g_* (wbench.sh is the reference)
GQ=$H/gpuq.sh; [ -e $GQ ] || GQ=$H/../gpuq.sh; GPUQ_LOCK=${LOCK:-}; . $GQ
HDR="tag	box	scroll	ids	ids0	ids_a	ids_b	ids_sd	e08	p05	p1	p2	p4	p05x	p1x	p2x	p4x	pmed	gross	gross100	conf	ceil	ceil_p1x	frac	pred_s	fit_s	bin	fcfg	cnt_bias	cnt_off	cnt_offp	dcnt_bias	dcnt_off	dcnt_offp"
[ -s $RES ] || echo "$HDR" > $RES
[ -s $RS ] || printf 'tag\tbox\tscroll\tids\tids_a\tids_b\tp1x\tp4x\tjoin\tjoined\tjoined_p4x\tfit_s\tbin\tfcfg\n' > $RS
[ -s $RR ] || printf 'tag\tsite\tslab\tr0_mm\tr1_mm\tvoxels\tm2\tm1\tzero\tp1\tp2\n' > $RR
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $O/run.log; }

if [ "$BOXES" = core ]; then BL=$(grep -v '^#' $BT | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1)
else BL=$BOXES; fi
step=$(head -c 16384 $CK | tr ',' '\n' | grep -a -m1 '"step":' | grep -aoE '[0-9]+')
log "wgate $TAG ckpt $CK (step ${step:-?}) vs $REF: $(echo $BL | wc -w) boxes, sites ${SITES:-none}, predict $(basename $UP), fitter $(basename $U) ($FCFG), lanes $FL"
echo "$CK" > $O/ckpt
rm -f $O/ready/.end; rm -rf $O/claim/*

gpredict() {   # gpredict like predict; SEAM=1: two passes, window grid at 0 and shifted 8 voxels on every axis (another phase of the
  # network's pooling against the 16^3 volcomp codec blocks), averaged by field-blend (phase as the complex mean)
  [ "$SEAM" = 1 ] || { predict "$@"; return; }
  local f=$1; shift
  GO=0,0,0 predict $f.g0 "$@" && GO=8,8,8 predict $f.g8 "${@:1:6}" ${7}.g8 ${8:-} || return 1
  RUNBIG $U field-blend $f $f.g0 $f.g8 > ${7}.blend 2>&1; local rc=$?; rm -rf $f.g0 $f.g8; return $rc
}
boxrow() {   # boxrow B: fits + row of one predicted box (a box lane)
  local B=$1 nm sc z y x nz ny nx lv note
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"; scroll $sc || return
  local d=$O/$B f=$FD/$B t1 t2 cz cy cx; mkdir -p $d; t1=$(date +%s)
  read -r cz cy cx <<< "$(jq -r '.shape | map(tostring) | join(" ")' $f/fields.json)"
  fitsc $f $d/fit $d/fit.log &
  if [ $REP -ge 1 ]; then bash $H/mkcrop.sh $f $d/fa 16 16 16 $((cz - 16)) $((cy - 16)) $((cx - 16)) > /dev/null && fitsc $d/fa $d/fita $d/fita.log & fi
  if [ $REP -ge 2 ]; then bash $H/mkcrop.sh $f $d/fb 0 0 0 $((cz - 16)) $((cy - 16)) $((cx - 16)) > /dev/null && fitsc $d/fb $d/fitb $d/fitb.log & fi
  wait
  # a fit that died (e.g. GPU memory beside the predicts and the slab lane) runs again alone, once
  grep -q 'winding ids right' $d/fit.log 2> /dev/null || { log "$B: fit retried"; fitsc $f $d/fit $d/fit.log; }
  [ $REP -ge 1 ] && ! grep -q 'winding ids right' $d/fita.log 2> /dev/null && { log "$B: fit a retried"; fitsc $d/fa $d/fita $d/fita.log; }
  [ $REP -ge 2 ] && ! grep -q 'winding ids right' $d/fitb.log 2> /dev/null && { log "$B: fit b retried"; fitsc $d/fb $d/fitb $d/fitb.log; }
  t2=$(date +%s)
  local ids0 ia=na ib=na ids sd e08 pp m pe=na gr=na g1=na cf=na cnb=na cno=na cnp=na dnb=na dno=na dnp=na cl clp fr
  ids0=$(num $(g_ids $d/fit.log)); [ $REP -ge 1 ] && ia=$(num $(g_ids $d/fita.log)); [ $REP -ge 2 ] && ib=$(num $(g_ids $d/fitb.log))
  rm -rf $d/fa $d/fb $d/fita $d/fitb
  read -r ids sd <<< "$(echo "$ids0 $ia $ib" | awk '{n=0; s=0; q=0; for (i=1;i<=3;i++) if ($i != "na") { n++; s+=$i; q+=$i*$i } if (!n) { print "na na"; exit } m=s/n; v=n>1 ? (q-n*m*m)/(n-1) : 0; printf "%.4f %.4f\n", m, sqrt(v > 0 ? v : 0)}')"
  e08=$(num $(g_e08 $d/fit.log))
  pp=""; for m in 0.5 1 2 4; do pp="$pp	$(num $(g_pair $d/fit.log $m 0))"; done; for m in 0.5 1 2 4; do pp="$pp	$(num $(g_pair $d/fit.log $m 1))"; done
  if [ "$DIAG" = 1 ]; then   # network phase vs label phase (CPU only)
    RUN $U fields-oracle $f $PR $PK $d/orc --fill none --phase-scale $PF > $d/orc.log 2>&1; RUN $U fit-compare $d/orc $f > $d/cmp.log 2>&1; rm -rf $d/orc
    pe=$(num $(grep -oE 'phase \|error\| \(turns\): median [0-9.]+' $d/cmp.log | awk '{print $NF}')); gr=$(num $(grep -oE '> 0.25 [0-9.]+' $d/cmp.log | head -1 | awk '{print $3}'))
    g1=$(num $(grep 'gross phase errors by reference spacing' $d/cmp.log | grep -oE '< 100 um [0-9.]+' | awk '{print $4}'))
    cf=$(num $(grep -oE 'phase confidence: median [0-9.]+' $d/cmp.log | awk '{print $NF}'))
    read -r cnb cno cnp <<< "$(g_cnt $d/cmp.log phase)"; read -r dnb dno dnp <<< "$(g_cnt $d/cmp.log density)"
  fi
  read -r cl clp <<< "$(awk -F'\t' -v b=$B 'NR > 1 && $2 == b {print $22, $23; exit}' $RREF)"; cl=${cl:-na}; clp=${clp:-na}   # the baseline's label ceiling
  fr=$(echo "$ids $cl" | awk '$1 != "na" && $2 != "na" && $2 > 0 {printf "%.4f", $1 / $2; exit} {print "na"}')
  echo "$TAG	$B	$sc	$ids	$ids0	$ia	$ib	$sd	$e08$pp	$pe	$gr	$g1	$cf	$cl	$clp	$fr	$(cat $O/ready/$B 2> /dev/null || echo na)	$((t2 - t1))	$BIN	$FCFG	$cnb	$cno	$cnp	$dnb	$dno	$dnp" >> $RES
  log "$B ids $ids (box $ids0 shifted $ia $ib) p05x $(echo "$pp" | cut -f6) fits $((t2 - t1)) s"
}
# claim B: one lane per box (a flock-guarded marker; plain mkdir claims raced in the 10-26 sandbox: two lanes both fitted PH1)
claim() { ( flock 7; [ -e $O/claim/$1 ] && exit 1; : > $O/claim/$1 ) 7> $O/claim.lock; }
boxlane() {   # boxes in list order as their fields land
  local B
  for B in $BL; do
    grep -qP "^$TAG\t$B\t" $RES && continue
    while [ ! -e $O/ready/$B ] && [ ! -e $O/ready/.end ]; do sleep 1; done
    [ -e $O/ready/$B ] || continue
    claim $B || continue
    boxrow $B
  done
}
# a slab-lane GPU step: under the run's GPU block while it lasts; once the block is released (boxes done) each step queues on its
# own (gpuq ${SPRIO:-bg}: other jobs get the GPU between slab fits)
gstep() { if [ -e $O/ready/.released ]; then ( GPUQ_HELD=0; gpuq_acquire ${SPRIO:-bg} wgate-$TAG-slab; "$@"; rc=$?; gpuq_release; exit $rc ); else "$@"; fi; }
sfit() {   # sfit LOG fit-args...: a slab fit, run again once if it died (10-26: one was OOM-killed in the shared ufsm.slice)
  local lg=$1 try; shift
  for try in 1 2; do UFSM_FIT_THREADS=$THREADS gstep RUNBIG $U fit "$@" > $lg 2>&1; grep -q 'winding ids right' $lg && return 0; log "$(basename $(dirname $lg)) $(basename $lg) died (try $try)"; done; return 1; }
slabsite() {   # slabsite s: wbench.sh's slab site (2 slabs, fit-join, joined score) on this run's site fields
  local s=$1 S=site$1 d f t0 t1 k a b m p1 p4 js; d=$O/$S; f=$FD/$S; mkdir -p $d; t0=$(date +%s); scroll paris4
  bash $H/mkcrop.sh $f $d/fA 0 96 96 96 3328 3520 > /dev/null; bash $H/mkcrop.sh $f $d/fB 64 96 96 96 3328 3520 > /dev/null; bash $H/mkcrop.sh $f $d/fC 0 96 96 160 3328 3520 > /dev/null
  for k in A B; do sfit $d/fit$k.log $d/f$k $GF --out $d/fit$k --codes $QR $QK --score-detail 1 --score-pairs 1 --dir 1 --uw-repair 8
    grep -E '^    radius' $d/fit$k.log | awk -v t=$TAG -v s=$S -v k=$k '{gsub(/-/, " "); $0 = $0; print t"\t"s"\t"k"\t"$2"\t"$3"\t"$5"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11}' >> $RR; done
  RUN $U fit-join $d/joined $d/fC --blend 16 $d/fitA 32 160 $d/fitB 160 288 > $d/join.log 2>&1
  js=$(num $(grep -oE '^ +0\|1 .*' $d/join.log | awk '{print $NF}'))
  sfit $d/joined.log $d/fC --model $d/joined $GF --codes $QR $QK --score-pairs 1
  t1=$(date +%s); a=$(num $(g_ids $d/fitA.log)); b=$(num $(g_ids $d/fitB.log))
  m=$(echo "$a $b" | awk '$1 != "na" && $2 != "na" {printf "%.4f", ($1 + $2) / 2; exit} {print "na"}')
  p1=$(echo "$(num $(g_pair $d/fitA.log 1 1)) $(num $(g_pair $d/fitB.log 1 1))" | awk '$1 != "na" && $2 != "na" {printf "%.4f", ($1 + $2) / 2; exit} {print "na"}')
  p4=$(echo "$(num $(g_pair $d/fitA.log 4 1)) $(num $(g_pair $d/fitB.log 4 1))" | awk '$1 != "na" && $2 != "na" {printf "%.4f", ($1 + $2) / 2; exit} {print "na"}')
  grep -q 'winding ids right' $d/joined.log || { log "$S FAILED (no joined score; no row: a rerun redoes the site)"; return; }
  echo "$TAG	$S	p4slab	$m	$a	$b	$p1	$p4	$js	$(num $(g_ids $d/joined.log))	$(num $(g_pair $d/joined.log 4 1))	$((t1 - t0))	$BIN	$FCFG" >> $RS
  log "$S slabs $a / $b join share $js joined $(g_ids $d/joined.log) ($((t1 - t0)) s)"
  rm -rf $d/fA $d/fB $d/fC $d/fitA $d/fitB $d/joined
}
slablane() { local s; for s in $SITES; do grep -qP "^$TAG\tsite$s\t" $RS && continue
    while [ ! -e $O/ready/site$s ] && [ ! -e $O/ready/.end ]; do sleep 2; done
    [ -e $O/ready/site$s ] && slabsite $s; done; }

T0=$(date +%s)
[ -n "${NOLOCK:-}" ] || gpuq_acquire ${PRIO:-eval} wgate-$TAG
TG=$(date +%s); [ $((TG - T0)) -gt 5 ] && log "GPU after $((TG - T0)) s in the queue"
rm -f $O/ready/.released
slablane & SL=$!; LANES=$SL; BLANES=
for i in $(seq $FL); do boxlane & BLANES="$BLANES $!"; done; LANES="$LANES $BLANES"
# the lanes hold the GPU lock too (inherited fd): on any exit of this script stop them (they finish the box at hand)
trap 'touch $O/ready/.end' EXIT; trap 'touch $O/ready/.end; kill $LANES 2> /dev/null; exit 143' INT TERM
# ---- the GPU lane: predicts only (sites first: the slab lane is the longest) ----
scroll paris4
for s in $SITES; do
  grep -qP "^$TAG\tsite$s\t" $RS && continue; f=$FD/site$s; mkdir -p $O/site$s
  if [ ! -e $f/fields.json ]; then t0=$(date +%s); gpredict $f $((2 * s)) 1024 512 320 7168 7680 $O/site$s/pred.log big || { log "site$s predict failed"; continue; }; log "site$s predicted in $(( $(date +%s) - t0 )) s"; fi
  echo 0 > $O/ready/site$s
done
for B in $BL; do
  grep -qP "^$TAG\t$B\t" $RES && continue
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"; [ -n "${nx:-}" ] || { log "no box $B"; continue; }
  scroll $sc || { log "$B: unknown scroll $sc"; continue; }
  f=$FD/$B; mkdir -p $O/$B; t0=$(date +%s)
  if [ ! -e $f/fields.json ]; then gpredict $f $z $y $x $nz $ny $nx $O/$B/pred.log || { log "$B predict failed"; continue; }; fi
  echo $(( $(date +%s) - t0 )) > $O/ready/$B
done
touch $O/ready/.end; TP=$(date +%s); log "GPU lane done: predicts $((TP - TG)) s"
wait $BLANES
# the block ends with the boxes: the remaining slab fits queue per fit (the slab lane is mostly CPU scoring at low GPU use)
if [ -z "${NOLOCK:-}" ]; then touch $O/ready/.released; gpuq_release; fi
log "boxes done: GPU block $(( $(date +%s) - TG )) s ($((TG - T0)) s queued); slab lane continues per fit"
wait $SL
log "wgate $TAG: $(( $(date +%s) - T0 )) s in all"
[ "$KEEP" = 0 ] && rm -rf $FD
bash $H/wgate-table.sh $TAG | tee $O/table.txt

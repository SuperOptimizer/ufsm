#!/usr/bin/env bash
# wbench.sh CKPT TAG: the winding + fitter bench (README.md beside this script).
#   For every box of boxes.tsv: predict fit fields (unless FIELDS=TAG0 reuses TAG0's fields: fixed-fields mode for fitter changes) ->
#   fit (production: --dir 1 --uw-repair 8, + FIT) scored vs the published-wrap codes (ids right, e08, gauge-free pair accuracy) on the
#   box and on REP shifted sub-domains (fit noise) -> network phase vs the label phase (fit-compare) -> the label-phase ceiling (the
#   same fitter on label phase; cached per fitter config). Then (SLABS=1) Paris 4 full-cross-section slab sites: 2 overlapping
#   96-cell slab fits per site, radius profile, fit-join overlap share, joined site score.
#   Rows: $WB/res/TAG.tsv (boxes), $WB/res/TAG-slabs.tsv, $WB/res/TAG-radius.tsv; logs $WB/runs/TAG/. Compare with ufsm wbench-stats.
# Env: SITE laptop|desk (default: laptop if ~/ufsm-data exists), GPU (0), BOXES ("clean" default: boxes not tagged label-noise in boxes.tsv | "core": the clean paris4 + paris4j boxes + 12 1667 boxes tagged core (routine snapshot evals) | "all" | "none" | names), REP (2: shifted fits per box, 0 off),
#   SLABS (1), SLABORC (none|net: slab ceiling, label phase in the slab crops; BOXES=none skips the boxes), SITES (z cells of the slab sites), FIT (extra fit flags; changes the fitter config tag; "{box}" becomes the box name,
#   e.g. FIT="--emb /vesuvius/ufsm/repr/emb/bench/{box} --uw-rep-emb 2"), JOIN (extra fit-join flags at the slab sites, "{box}" = site<s>, e.g.
#   JOIN="--fibers /vesuvius/ufsm/fibers/wbsites/{box}.fib --join-fib 1"; changes the fitter config tag), FIELDS (TAG0: reuse fields),
#   KEEP (1: keep this TAG's fields for later fixed-fields runs; default 1), U (binary; default $WB/bin/ufsm = the pinned build), UP (predict binary, default U: a newer build for checkpoints the fitter pin cannot load),
#   NOLOCK=1 (the caller holds the GPU lock).
#   WZARGS= (empty): cube windows, no --window-z (latent-grid checkpoints: train --lat 4), with WIN / HALO.
#   LATENTS="SCROLL=STORE ..." (e.g. "paris4=/mnt/scratch/lat/p4-l2.lc 1667=/mnt/scratch/lat/p1667-l2b.lc"): those scrolls' fields predicted from
#   the ufsm lc-encode store of their CT level (predict --latents: --lat-basis checkpoints; paris4j falls back to paris4's store).
#   PBATCH (1): the boxes' fields predicted first, one predict --boxes process per scroll (needs a predict binary with --boxes, 2026-11-12:
#   UP / U older than that: PBATCH=0); 0: a predict process per box inside the box loop, as before.
set -u
H=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE:-$([ -d $HOME/ufsm-data ] && echo laptop || echo desk)}; GPU=${GPU:-0}
. $H/site-$SITE.env
CK=$(readlink -f "$1"); TAG=$2
U=${U:-$WB/bin/ufsm}; [ -x $U ] || { echo "no binary $U"; exit 2; }
BOXES=${BOXES:-clean}; SLABORC=${SLABORC:-}; REP=${REP:-2}; SLABS=${SLABS:-$([ "$SITE" = laptop ] && echo 0 || echo 1)}; SITES=${SITES:-4624 5392 5904 6672}; FIT=${FIT:-}; JOIN=${JOIN:-}; FIELDS=${FIELDS:-}; KEEP=${KEEP:-1}
[ "$SITE" = laptop ] && [ "$SLABS" = 1 ] && { echo "wbench: no slab fits on the laptop (OOM in ufsm.slice): run the sites on forlindesk (wgate-slabs.sh) or SLABS=0"; exit 2; }
BT=${BT:-$H/boxes.tsv}; BIN=$(sha1sum $U | cut -c1-8)
GF="--gpu $GPU"; [ "${CPU:-0}" = 1 ] && GF="--cpu --threads $THREADS"   # CPU=1: fits on the CPU backend (no GPU; a separate fitter config)
FCFG=$(echo "$BIN --dir 1 --uw-repair 8 $FIT${CPU:+ cpu$CPU}${JOIN:+ join $JOIN}" | sha1sum | cut -c1-8)   # the fitter config: binary + flags
FT=${FIELDS:-$TAG}; FD=$WB/fields/$FT; O=$WB/runs/$TAG; mkdir -p $O $WB/res $WB/ceil $FD $CACHE
RES=$WB/res/$TAG.tsv; RS=$WB/res/$TAG-slabs.tsv; RR=$WB/res/$TAG-radius.tsv; CEIL=$WB/ceil/$FCFG.tsv
HDR="tag	box	scroll	ids	ids0	ids_a	ids_b	ids_sd	e08	p05	p1	p2	p4	p05x	p1x	p2x	p4x	pmed	gross	gross100	conf	ceil	ceil_p1x	frac	pred_s	fit_s	bin	fcfg	cnt_bias	cnt_off	cnt_offp	dcnt_bias	dcnt_off	dcnt_offp"
[ -s $RES ] || echo "$HDR" > $RES
[ -s $CEIL ] || printf 'box\tceil\tceil_p1x\tfrom\n' > $CEIL
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $O/run.log; }
# GPU lock per box / per slab GPU step (released in between so short jobs interleave); NOLOCK=1: the caller holds it
LK=; [ -z "${NOLOCK:-}" ] && [ -n "${LOCK:-}" ] && [ "${CPU:-0}" != 1 ] && LK=$LOCK
# through the gpuq priority queue (scripts/gpuq.sh; PRIO bg by default: at every box / slab step it lets waiting jobs in, a "now"
# check first); plain flock users of the same lock stay excluded
GQ=$H/gpuq.sh; [ -e $GQ ] || GQ=$H/../gpuq.sh; GPUQ_LOCK=$LOCK; . $GQ
lk() { [ -n "$LK" ] && gpuq_acquire ${PRIO:-bg} wbench-$TAG; return 0; }
ulk() { [ -n "$LK" ] && gpuq_yield; return 0; }
log "wbench $TAG ckpt $CK fields $FT fitter $FCFG ($BIN $FIT) site $SITE gpu $GPU"
echo "$CK" > $O/ckpt; echo "$BIN $FIT" > $O/fitter

scroll() {   # scroll SCROLL: R V UM LV AX QR QK PR PK PF (PF: CT voxels per label-phase raster voxel)
  case $1 in
    paris4) R=$P4ROOT; V=${P4V:-PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr}; UM=2.4; LV=2; AX=$AXP4
            QR=$GT/paris4-q9p6/q-9p6.zarr; QK=9.6; PR=$GT/paris4-wind-20261019; PK=phaseA/9.6; PF=1;;
    paris4j) R=$P4ROOT; V=${P4V:-PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr}; UM=2.4; LV=2; AX=$AXP4   # July-only labels (codes rebuilt from the published July wraps)
            QR=$P4JQ; QK=9.6; PR=$GT/paris4-wind-20261019; PK=phaseB/9.6; PF=1;;
    1667)   R=$MIRROR; V=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr; UM=2.399; LV=2; AX=$AX1667
            QR=$GT/PHerc1667-wind-20261016/codes; QK=4.798; PR=$GT/PHerc1667-wind-20261016; PK=phase/9.6; PF=1;;
    PHerc0139) R=$MIRROR; V=PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr; UM=9.362; LV=0; AX=$GT/axis/PHerc0139-umb-scan.json
            QR=$GT/PHerc0139-wind-20261016/codes; QK=18.72; PR=$GT/PHerc0139-wind-20261016/phase; PK=18.72; PF=2;;
    PHerc0814) R=$MIRROR; V=PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr; UM=9.362; LV=0; AX=$GT/axis/PHerc0814-20250804134230.json
            QR=$GT/PHerc0814-wind-20261019/codes; QK=18.724; PR=$GT/PHerc0814-wind-20261019/phase; PK=18.724; PF=2;;
    PHerc0500P2) R=$MIRROR; V=PHerc0500P2/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr; UM=9.362; LV=0; AX=$GT/axis/PHerc0500P2-20250820143440.json
            QR=$GT/PHerc0500P2-wind-20261019/codes; QK=18.724; PR=$GT/PHerc0500P2-wind-20261019/phase; PK=18.724; PF=2;;
    # cross-res (scripts/univ/xres_prep.sh, BT=$WB/xres/boxes-xres.tsv): the 9.36 um GT boxes on the registered 2.2-2.4 um scans, labels warped to level 3
    xPHerc0139) R=$MIRROR; V=PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr; UM=2.399; LV=2; AX=$WB/xres/PHerc0139/axis.json
            QR=$WB/xres/PHerc0139/codes; QK=19.192; PR=$WB/xres/PHerc0139/phase; PK=19.192; PF=2;;
    xPHerc0814) R=$MIRROR; V=PHerc0814/volumes/20260309142202-2.399um-0.2m-78keV-masked.zarr; UM=2.399; LV=2; AX=$WB/xres/PHerc0814/axis.json
            QR=$WB/xres/PHerc0814/codes; QK=19.192; PR=$WB/xres/PHerc0814/phase; PK=19.192; PF=2;;
    xPHerc0500P2) R=$MIRROR; V=PHerc0500P2/volumes/20250526151718-2.215um-0.4m-111keV-masked.zarr; UM=2.215; LV=2; AX=$WB/xres/PHerc0500P2/axis.json
            QR=$WB/xres/PHerc0500P2/codes; QK=17.72; PR=$WB/xres/PHerc0500P2/phase; PK=17.72; PF=2;;
    *) return 1;;
  esac
}
lat_for() {   # lat_for SCROLL: its latent store in LATENTS (paris4j: paris4's), or nothing
  local kv; for kv in ${LATENTS:-}; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done
  [ "$1" = paris4j ] && lat_for paris4; return 0; }
predict() {   # predict OUTFIELDS z y x nz ny nx LOG [big]: fit fields of the box (production window, 19.2 / 18.7 um cells, lossless phase)
  local f=$1 lg=$8 run=RUN lx; [ "${9:-}" = big ] && run=RUNBIG; rm -rf $f $f.tmp
  lx=$(lat_for ${sc:-}); [ -n "${LATENTS:-}" ] && [ -z "$lx" ] && { echo "wbench: LATENTS has no store for scroll ${sc:-?}" > $lg; return 1; }
  # DLV (resolution test, 2026-11-11): predict at level LV+DLV (DLV -1: 4.8 um input for the 9.6 um boxes, -2: 2.4 um, +1: 19.2 um) with
  # the box in that level's voxels and --fit-h 2^(1-DLV), so the fit fields keep their 19.2 um cells and the fitter / codes are unchanged.
  pbox $2 $3 $4 $5 $6 $7 $lg || return 1
  local rr vv; ctpick $lv $b0,$b1,$b2,$b3,$b4,$b5; echo "ct: $rr $vv level $lv" > $lg.root
  $run ${UP:-$U} predict $CK $rr $vv $f.tmp --box $b0,$b1,$b2,$b3,$b4,$b5 --fit-zarr $f $(pargs) ${lx:+--latents $lx} > $lg 2>&1; local rc=$?; rm -rf $f.tmp; return $rc
}
ctpick() {   # ctpick LV b0,..,b5 [box ...]: rr vv = the local CT copy (LOCALCT, the desk, 2026-11-12; keys without volumes/) when ctcover.py finds
             # every shard the predict can touch there (or absent from the mirror too: air), else R V. Same bytes either way (one codec).
  local l=$1; shift; rr=$R vv=$V; [ -n "${LOCALCT:-}" ] && [ "$R" != "$LOCALCT" ] || return 0
  local lk_=${V/\/volumes\//\/} bx; for bx in "$@"; do python3 -I $H/ctcover.py $LOCALCT $lk_ $R $V $l $bx > /dev/null 2>> $O/ctcover.log || return 0; done
  rr=$LOCALCT vv=$lk_; }
pbox() {   # pbox z y x nz ny nx LOG: the box in the predict level's voxels (b0..b5), that level (lv) and --fit-h (fh)
  # DLV (resolution test, 2026-11-11): predict at level LV+DLV (DLV -1: 4.8 um input for the 9.6 um boxes, -2: 2.4 um, +1: 19.2 um) with
  # the box in that level's voxels and --fit-h 2^(1-DLV), so the fit fields keep their 19.2 um cells and the fitter / codes are unchanged.
  local dl=${DLV:-0}; lv=$LV b0=$1 b1=$2 b2=$3 b3=$4 b4=$5 b5=$6 fh=2
  if [ $dl != 0 ]; then lv=$((LV + dl)); [ $lv -ge 0 ] || { echo "wbench: DLV $dl below level 0 for scroll ${sc:-?}" > $7; return 1; }
    if [ $dl -lt 0 ]; then local m=$((1 << -dl)); b0=$((b0 * m)) b1=$((b1 * m)) b2=$((b2 * m)) b3=$((b3 * m)) b4=$((b4 * m)) b5=$((b5 * m)); fh=$((2 * m))
    else local m=$((1 << dl)); b0=$((b0 / m)) b1=$((b1 / m)) b2=$((b2 / m)) b3=$((b3 / m)) b4=$((b4 / m)) b5=$((b5 / m)); fh=$((2 / m)); fi
    [ $fh -ge 1 ] || { echo "wbench: DLV $dl leaves no fit cell" > $7; return 1; }; fi
}
pargs() {   # the predict options every box of a scroll shares (after pbox: lv, fh)
  echo --um $UM --level $lv ${WZARGS---window-z ${WINZ:-384} --halo-z ${HALOZ:-64}} --window ${WIN:-640} --halo ${HALO:-64} --shard 512 \
    --grid-origin ${GO:-0,0,0} --ema 1 --q 8 --threads $THREADS --gpu $GPU --axis $AX --levels 1 --fit-h $fh --fit-q 8 --fit-q-phase 0 ${PX:-} --no-out 1 --cache $CACHE
}
pbatch() {   # PBATCH=1 (default): the fields of every box still to run, one predict --boxes process per scroll (the ~5 s start-up of a predict run
             # paid once per scroll, not per box; the same bytes as a process per box); per box its seconds in $O/BOX/pred.s
  local B; for B in $BL; do [ -e $FD/$B/fields.json ] || rm -f $O/$B/pred.s; done   # (a pred.s is this run's only, or that of the PHASE=predict run whose fields these are)
  [ -n "$FIELDS" ] || [ "${PBATCH:-1}" != 1 ] && return 0
  [ -n "${STOPAT:-}" ] && [ $(date +%s) -ge $STOPAT ] && return 0   # (the window is over: nothing new on the GPU)
  local s nm z y x nz ny nx lv_ note n bf lg lx
  for s in $(for B in $BL; do awk -F'\t' -v b=$B '$1 == b {print $2}' $BT; done | sort -u); do
    sc=$s; scroll $sc || continue
    lx=$(lat_for $sc); [ -n "${LATENTS:-}" ] && [ -z "$lx" ] && continue   # (predict() reports it per box)
    bf=$O/boxes-$sc.txt; lg=$O/pred-$sc.log; : > $bf; n=0
    for B in $BL; do
      IFS=$'\t' read -r nm s2 z y x nz ny nx lv_ note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"
      [ "${s2:-}" = $sc ] || continue
      grep -qP "^$TAG\t$B\t" $RES && continue; [ -e $FD/$B/fields.json ] && continue
      mkdir -p $O/$B; pbox $z $y $x $nz $ny $nx $O/$B/pred.log || continue
      rm -rf $FD/$B $FD/$B.tmp; echo "$b0,$b1,$b2,$b3,$b4,$b5 $FD/$B.tmp --fit-zarr $FD/$B" >> $bf; echo $B >> $bf.names; n=$((n + 1))
    done
    [ $n = 0 ] && { rm -f $bf $bf.names; continue; }
    local rr vv; ctpick $lv $(cut -d' ' -f1 $bf); log "$sc: $n boxes predicted in one process ($bf; ct $rr $vv)"; lk
    RUN ${UP:-$U} predict $CK $rr $vv - --boxes $bf $(pargs) ${lx:+--latents $lx} > $lg 2>&1 || log "$sc: predict --boxes failed ($lg; the boxes left run one by one)"
    ulk
    local i=0; while read -r B; do i=$((i + 1)); rm -rf $FD/$B.tmp
      grep -oE "predict --boxes: box $i done in [0-9.]+" $lg | awk '{print int($NF + 0.5)}' > $O/$B/pred.s; cp $lg $O/$B/pred.log; done < $bf.names
    rm -f $bf.names
  done
}
fitsc() {   # fitsc FIELDS OUT LOG [extra]: production fit + codes score + pair accuracy
  local f=$1 o=$2 lg=$3; shift 3
  UFSM_FIT_THREADS=$THREADS RUN $U fit $f $GF --out $o --codes $QR $QK --dir 1 --uw-repair 8 --score-pairs 1 ${FIT//\{box\}/$B} "$@" > $lg 2>&1
}
g_ids() { grep -oE 'winding ids right [0-9.]+' $1 | tail -1 | awk '{print $4}'; }
g_e08() { grep -oE '< 0.08 turn [0-9.]+' $1 | tail -1 | awk '{print $4}'; }
g_pair() { grep -oE "pair winding right.*" $1 | grep -oE "$2 mm [0-9.na]+ cross [0-9.na]+" | awk -v c=$3 '{print (c ? $5 : $3)}'; }
num() { [[ "${1:-}" =~ ^[0-9.]+$ ]] && echo $1 || echo na; }
g_cnt() {   # fit-compare turn counts on 0.5 mm radial segments, by phase | density: median error (turns), share >= 0.5 turn off, the same on packed segments
  local l; l=$(grep -E "^ +by $2:" $1 | head -1); [ -n "$l" ] || { echo "na na na"; return; }
  echo "$(echo "$l" | grep -oE "median [-+]?[0-9.]+" | awk "{print \$2+0}") $(echo "$l" | grep -oE "0.5 turn [0-9.]+" | awk "{print \$3}") $(echo "$l" | grep -oE "um: [0-9.]+" | awk "{print \$2}")"; }

# ---- boxes ----
if [ "$BOXES" = all ]; then BL=$(grep -v '^#' $BT | cut -f1)
elif [ "$BOXES" = none ]; then BL=""
elif [ "$BOXES" = clean ]; then BL=$(grep -v '^#' $BT | awk -F'\t' '$10 !~ /label-noise/' | cut -f1)
elif [ "$BOXES" = core ]; then BL=$(grep -v '^#' $BT | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1)   # routine snapshot evals
else BL=$BOXES; fi
pbatch
for B in $BL; do
  ulk
  [ -n "${STOPAT:-}" ] && [ $(date +%s) -ge $STOPAT ] && { log "stop: the GPU window ends (STOPAT); $B and later boxes left for the next window"; break; }
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"; [ -n "${nx:-}" ] || { log "no box $B"; continue; }
  grep -qP "^$TAG\t$B\t" $RES && { log "$B done"; continue; }
  scroll $sc || { log "$B: unknown scroll $sc"; continue; }
  d=$O/$B; mkdir -p $d; f=$FD/$B
  lk; t0=$(date +%s)   # (pred_s without the GPU queue wait)
  if [ -n "$FIELDS" ]; then [ -e $f/fields.json ] || { log "$B: no fields $f"; continue; }
  elif [ ! -e $f/fields.json ]; then rm -f $d/pred.s; predict $f $z $y $x $nz $ny $nx $d/pred.log || { log "$B predict failed"; continue; }; fi
  # PHASE=predict (2026-11-12): only the predicts of the row (fields stay on disk); a later plain run fits and scores from them
  [ "${PHASE:-}" = predict ] && continue
  t1=$(date +%s); ps=$((t1 - t0)); [ -s $d/pred.s ] && ps=$(cat $d/pred.s)   # (pbatch: the box's own seconds in the --boxes run)
  fitsc $f $d/fit $d/fit.log; t2=$(date +%s)
  ids0=$(num $(g_ids $d/fit.log))
  # REP shifted sub-domains: drop the first / last 16 cells (32 level voxels) of every axis
  ia=na; ib=na
  if [ $REP -ge 1 ]; then
    read -r cz cy cx <<< "$(jq -r '.shape | map(tostring) | join(" ")' $f/fields.json)"
    bash $H/mkcrop.sh $f $d/fa 16 16 16 $((cz - 16)) $((cy - 16)) $((cx - 16)) > /dev/null && fitsc $d/fa $d/fita $d/fita.log && ia=$(num $(g_ids $d/fita.log))
    [ $REP -ge 2 ] && bash $H/mkcrop.sh $f $d/fb 0 0 0 $((cz - 16)) $((cy - 16)) $((cx - 16)) > /dev/null && fitsc $d/fb $d/fitb $d/fitb.log && ib=$(num $(g_ids $d/fitb.log))
    rm -rf $d/fa $d/fb $d/fita $d/fitb
  fi
  read -r ids sd <<< "$(echo "$ids0 $ia $ib" | awk '{n=0; s=0; q=0; for (i=1;i<=3;i++) if ($i != "na") { n++; s+=$i; q+=$i*$i } if (!n) { print "na na"; exit } m=s/n; v=n>1 ? (q-n*m*m)/(n-1) : 0; printf "%.4f %.4f\n", m, sqrt(v > 0 ? v : 0)}')"
  e08=$(num $(g_e08 $d/fit.log))
  pp=""; for m in 0.5 1 2 4; do pp="$pp	$(num $(g_pair $d/fit.log $m 0))"; done; for m in 0.5 1 2 4; do pp="$pp	$(num $(g_pair $d/fit.log $m 1))"; done
  # network phase vs label phase (label spacing as sp)
  RUN $U fields-oracle $f $PR $PK $d/orc --fill none --phase-scale $PF > $d/orc.log 2>&1; RUN $U fit-compare $d/orc $f > $d/cmp.log 2>&1
  pe=$(num $(grep -oE 'phase \|error\| \(turns\): median [0-9.]+' $d/cmp.log | awk '{print $NF}')); gr=$(num $(grep -oE '> 0.25 [0-9.]+' $d/cmp.log | head -1 | awk '{print $3}'))
  g1=$(num $(grep 'gross phase errors by reference spacing' $d/cmp.log | grep -oE '< 100 um [0-9.]+' | awk '{print $4}'))
  cf=$(num $(grep -oE 'phase confidence: median [0-9.]+' $d/cmp.log | awk '{print $NF}'))
  read -r cnb cno cnp <<< "$(g_cnt $d/cmp.log phase)"; read -r dnb dno dnp <<< "$(g_cnt $d/cmp.log density)"
  rm -rf $d/orc
  # the ceiling of this fitter config (label phase where known, this run's network planes elsewhere); cached per box
  if ! grep -qP "^$B\t" $CEIL; then
    RUN $U fields-oracle $f $PR $PK $d/swap --fill net --phase-scale $PF > $d/swap.log 2>&1 && fitsc $d/swap $d/fitswap $d/fitswap.log
    printf '%s\t%s\t%s\t%s\n' $B $(num $(g_ids $d/fitswap.log)) $(num $(g_pair $d/fitswap.log 1 1)) $FT >> $CEIL; rm -rf $d/swap $d/fitswap
  fi
  read -r cl clp <<< "$(awk -F'\t' -v b=$B '$1 == b {print $2, $3}' $CEIL | tail -1)"
  fr=$(echo "$ids $cl" | awk '$1 != "na" && $2 != "na" && $2 > 0 {printf "%.4f", $1 / $2; exit} {print "na"}')
  echo "$TAG	$B	$sc	$ids	$ids0	$ia	$ib	$sd	$e08$pp	$pe	$gr	$g1	$cf	$cl	$clp	$fr	$ps	$((t2 - t1))	$BIN	$FCFG	$cnb	$cno	$cnp	$dnb	$dno	$dnp" >> $RES
  log "$B ids $ids (box $ids0 shifted $ia $ib) p1x $(g_pair $d/fit.log 1 1) gross100 $g1 cnt_bias $cnb cnt_off $cno ceil $cl pred ${ps}s fit $((t2 - t1))s"
  # (the fit dir is kept: others read runs/TAG/BOX/fit)
  [ -z "$FIELDS" ] && [ "$KEEP" = 0 ] && rm -rf $f
done
ulk

# ---- Paris 4 full-cross-section slab sites (in-sample for every Paris 4-trained model) ----
# site s (19.2 um field cells, s = 256 k: the box on the predict window / shard grid): fields cells [s, s+160) x y 512..4096 x 256..4096,
# slabs cropped to the production union crop y 608..3936 x 352..3872 (field cells); slab A = cells
# [0, 96), B = [64, 160) of the site (production 64-cell interiors + 16 overlap), A/B interiors [16, 80) / [80, 144) joined by fit-join.
if [ "$SLABS" = 1 ]; then
  scroll paris4
  [ -s $RS ] || printf 'tag\tbox\tscroll\tids\tids_a\tids_b\tp1x\tp4x\tjoin\tjoined\tjoined_p4x\tfit_s\tbin\tfcfg\n' > $RS
  [ -s $RR ] || printf 'tag\tsite\tslab\tr0_mm\tr1_mm\tvoxels\tm2\tm1\tzero\tp1\tp2\n' > $RR
  for s in $SITES; do
    ulk
    S=site$s; grep -qP "^$TAG\t$S\t" $RS && { log "$S done"; continue; }
    d=$O/$S; mkdir -p $d; f=$FD/$S; t0=$(date +%s)
    if [ -n "$FIELDS" ]; then [ -e $f/fields.json ] || { log "$S: no fields $f"; continue; }
    elif [ ! -e $f/fields.json ]; then lk; predict $f $((2 * s)) 1024 512 320 7168 7680 $d/pred.log big || { log "$S predict failed"; continue; }; ulk; fi
    bash $H/mkcrop.sh $f $d/fA 0 96 96 96 3328 3520 > /dev/null; bash $H/mkcrop.sh $f $d/fB 64 96 96 96 3328 3520 > /dev/null; bash $H/mkcrop.sh $f $d/fC 0 96 96 160 3328 3520 > /dev/null
    if [ -n "$SLABORC" ]; then   # slab ceiling: label phase in the slab crops (--fill net: network planes where unlabelled)
      for k in A B C; do RUNBIG $U fields-oracle $d/f$k $PR $PK $d/o$k --fill $SLABORC --phase-scale $PF > $d/orc$k.log 2>&1 || { log "$S: fields-oracle $k failed"; continue 2; }
        rm -rf $d/f$k; mv $d/o$k $d/f$k; done
    fi
    for k in A B; do lk; UFSM_FIT_THREADS=$THREADS RUNBIG $U fit $d/f$k $GF --out $d/fit$k --codes $QR $QK --score-detail 1 --score-pairs 1 --dir 1 --uw-repair 8 ${FIT//\{box\}/$S} > $d/fit$k.log 2>&1; ulk
      grep -E '^    radius' $d/fit$k.log | awk -v t=$TAG -v s=$S -v k=$k '{gsub(/-/, " "); $0 = $0; print t"\t"s"\t"k"\t"$2"\t"$3"\t"$5"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11}' >> $RR; done
    RUN $U fit-join $d/joined $d/fC --blend 16 ${JOIN//\{box\}/$S} $d/fitA 32 160 $d/fitB 160 288 > $d/join.log 2>&1
    js=$(num $(grep -oE '^ +0\|1 .*' $d/join.log | awk '{print $NF}'))
    lk; UFSM_FIT_THREADS=$THREADS RUNBIG $U fit $d/fC --model $d/joined $GF --codes $QR $QK --score-pairs 1 > $d/joined.log 2>&1; ulk
    t1=$(date +%s); a=$(num $(g_ids $d/fitA.log)); b=$(num $(g_ids $d/fitB.log))
    m=$(echo "$a $b" | awk '$1 != "na" && $2 != "na" {printf "%.4f", ($1 + $2) / 2; exit} {print "na"}')
    p1=$(echo "$(num $(g_pair $d/fitA.log 1 1)) $(num $(g_pair $d/fitB.log 1 1))" | awk '$1 != "na" && $2 != "na" {printf "%.4f", ($1 + $2) / 2; exit} {print "na"}')
    p4=$(echo "$(num $(g_pair $d/fitA.log 4 1)) $(num $(g_pair $d/fitB.log 4 1))" | awk '$1 != "na" && $2 != "na" {printf "%.4f", ($1 + $2) / 2; exit} {print "na"}')
    echo "$TAG	$S	p4slab	$m	$a	$b	$p1	$p4	$js	$(num $(g_ids $d/joined.log))	$(num $(g_pair $d/joined.log 4 1))	$((t1 - t0))	$BIN	$FCFG" >> $RS
    log "$S slabs $a / $b join share $js joined $(g_ids $d/joined.log) ($((t1 - t0)) s)"
    rm -rf $d/fA $d/fB $d/fC $d/fitA $d/fitB $d/joined
    [ -z "$FIELDS" ] && [ "$KEEP" = 0 ] && rm -rf $f
  done
fi
[ -n "$LK" ] && gpuq_release
log "wbench $TAG finished"

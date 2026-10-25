#!/usr/bin/env bash
# wbench.sh CKPT TAG: the winding + fitter bench (README.md beside this script).
#   For every box of boxes.tsv: predict fit fields (unless FIELDS=TAG0 reuses TAG0's fields: fixed-fields mode for fitter changes) ->
#   fit (production: --dir 1 --uw-repair 8, + FIT) scored vs the published-wrap codes (ids right, e08, gauge-free pair accuracy) on the
#   box and on REP shifted sub-domains (fit noise) -> network phase vs the label phase (fit-compare) -> the label-phase ceiling (the
#   same fitter on label phase; cached per fitter config). Then (SLABS=1) Paris 4 full-cross-section slab sites: 2 overlapping
#   96-cell slab fits per site, radius profile, fit-join overlap share, joined site score.
#   Rows: $WB/res/TAG.tsv (boxes), $WB/res/TAG-slabs.tsv, $WB/res/TAG-radius.tsv; logs $WB/runs/TAG/. Compare with ufsm wbench-stats.
# Env: SITE laptop|desk (default: laptop if ~/ufsm-data exists), GPU (0), BOXES ("clean" default: boxes not tagged label-noise in boxes.tsv | "all" | names), REP (2: shifted fits per box, 0 off),
#   SLABS (1), SITES (z cells of the slab sites), FIT (extra fit flags; changes the fitter config tag; "{box}" becomes the box name,
#   e.g. FIT="--emb /vesuvius/ufsm/repr/emb/bench/{box} --uw-rep-emb 2"), FIELDS (TAG0: reuse fields),
#   KEEP (1: keep this TAG's fields for later fixed-fields runs; default 1), U (binary; default $WB/bin/ufsm = the pinned build),
#   NOLOCK=1 (the caller holds the GPU lock).
set -u
H=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE:-$([ -d $HOME/ufsm-data ] && echo laptop || echo desk)}; GPU=${GPU:-0}
. $H/site-$SITE.env
CK=$(readlink -f "$1"); TAG=$2
U=${U:-$WB/bin/ufsm}; [ -x $U ] || { echo "no binary $U"; exit 2; }
BOXES=${BOXES:-clean}; REP=${REP:-2}; SLABS=${SLABS:-1}; SITES=${SITES:-4624 5392 5904 6672}; FIT=${FIT:-}; FIELDS=${FIELDS:-}; KEEP=${KEEP:-1}
BT=${BT:-$H/boxes.tsv}; BIN=$(sha1sum $U | cut -c1-8)
FCFG=$(echo "$BIN --dir 1 --uw-repair 8 $FIT" | sha1sum | cut -c1-8)   # the fitter config: binary + flags
FT=${FIELDS:-$TAG}; FD=$WB/fields/$FT; O=$WB/runs/$TAG; mkdir -p $O $WB/res $WB/ceil $FD $CACHE
RES=$WB/res/$TAG.tsv; RS=$WB/res/$TAG-slabs.tsv; RR=$WB/res/$TAG-radius.tsv; CEIL=$WB/ceil/$FCFG.tsv
HDR="tag	box	scroll	ids	ids0	ids_a	ids_b	ids_sd	e08	p05	p1	p2	p4	p05x	p1x	p2x	p4x	pmed	gross	gross100	conf	ceil	ceil_p1x	frac	pred_s	fit_s	bin	fcfg"
[ -s $RES ] || echo "$HDR" > $RES
[ -s $CEIL ] || printf 'box\tceil\tceil_p1x\tfrom\n' > $CEIL
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $O/run.log; }
if [ -z "${NOLOCK:-}" ] && [ -n "${LOCK:-}" ]; then exec 9> $LOCK; log "waiting for $LOCK"; flock 9; fi
log "wbench $TAG ckpt $CK fields $FT fitter $FCFG ($BIN $FIT) site $SITE gpu $GPU"
echo "$CK" > $O/ckpt; echo "$BIN $FIT" > $O/fitter

scroll() {   # scroll SCROLL: R V UM LV AX QR QK PR PK PF (PF: CT voxels per label-phase raster voxel)
  case $1 in
    paris4) R=$P4ROOT; V=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; UM=2.4; LV=2; AX=$AXP4
            QR=$GT/paris4-q9p6/q-9p6.zarr; QK=9.6; PR=$GT/paris4-wind-20261019; PK=phaseA/9.6; PF=1;;
    paris4j) R=$P4ROOT; V=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; UM=2.4; LV=2; AX=$AXP4   # July-only labels (codes rebuilt from the published July wraps)
            QR=$P4JQ; QK=9.6; PR=$GT/paris4-wind-20261019; PK=phaseB/9.6; PF=1;;
    1667)   R=$MIRROR; V=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr; UM=2.399; LV=2; AX=$AX1667
            QR=$GT/PHerc1667-wind-20261016/codes; QK=4.798; PR=$GT/PHerc1667-wind-20261016; PK=phase/9.6; PF=1;;
    PHerc0139) R=$MIRROR; V=PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr; UM=9.362; LV=0; AX=$GT/axis/PHerc0139-umb-scan.json
            QR=$GT/PHerc0139-wind-20261016/codes; QK=18.72; PR=$GT/PHerc0139-wind-20261016/phase; PK=18.72; PF=2;;
    PHerc0814) R=$MIRROR; V=PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr; UM=9.362; LV=0; AX=$GT/axis/PHerc0814-20250804134230.json
            QR=$GT/PHerc0814-wind-20261019/codes; QK=18.724; PR=$GT/PHerc0814-wind-20261019/phase; PK=18.724; PF=2;;
    PHerc0500P2) R=$MIRROR; V=PHerc0500P2/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr; UM=9.362; LV=0; AX=$GT/axis/PHerc0500P2-20250820143440.json
            QR=$GT/PHerc0500P2-wind-20261019/codes; QK=18.724; PR=$GT/PHerc0500P2-wind-20261019/phase; PK=18.724; PF=2;;
    *) return 1;;
  esac
}
predict() {   # predict OUTFIELDS z y x nz ny nx LOG [big]: fit fields of the box (production window, 19.2 / 18.7 um cells, lossless phase)
  local f=$1 lg=$8 run=RUN; [ "${9:-}" = big ] && run=RUNBIG; rm -rf $f $f.tmp
  $run $U predict $CK $R $V $f.tmp --um $UM --level $LV --box $2,$3,$4,$5,$6,$7 --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads $THREADS --gpu $GPU --axis $AX --levels 1 --fit-zarr $f --fit-h 2 --fit-q 8 --fit-q-phase 0 \
    --no-out 1 --cache $CACHE > $lg 2>&1; local rc=$?; rm -rf $f.tmp; return $rc
}
fitsc() {   # fitsc FIELDS OUT LOG [extra]: production fit + codes score + pair accuracy
  local f=$1 o=$2 lg=$3; shift 3
  UFSM_FIT_THREADS=$THREADS RUN $U fit $f --gpu $GPU --out $o --codes $QR $QK --dir 1 --uw-repair 8 --score-pairs 1 ${FIT//\{box\}/$B} "$@" > $lg 2>&1
}
g_ids() { grep -oE 'winding ids right [0-9.]+' $1 | tail -1 | awk '{print $4}'; }
g_e08() { grep -oE '< 0.08 turn [0-9.]+' $1 | tail -1 | awk '{print $4}'; }
g_pair() { grep -oE "pair winding right.*" $1 | grep -oE "$2 mm [0-9.na]+ cross [0-9.na]+" | awk -v c=$3 '{print (c ? $5 : $3)}'; }
num() { [[ "${1:-}" =~ ^[0-9.]+$ ]] && echo $1 || echo na; }

# ---- boxes ----
if [ "$BOXES" = all ]; then BL=$(grep -v '^#' $BT | cut -f1)
elif [ "$BOXES" = clean ]; then BL=$(grep -v '^#' $BT | awk -F'\t' '$10 !~ /label-noise/' | cut -f1)
else BL=$BOXES; fi
for B in $BL; do
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"; [ -n "${nx:-}" ] || { log "no box $B"; continue; }
  grep -qP "^$TAG\t$B\t" $RES && { log "$B done"; continue; }
  scroll $sc || { log "$B: unknown scroll $sc"; continue; }
  d=$O/$B; mkdir -p $d; f=$FD/$B; t0=$(date +%s)
  if [ -n "$FIELDS" ]; then [ -e $f/fields.json ] || { log "$B: no fields $f"; continue; }
  elif [ ! -e $f/fields.json ]; then predict $f $z $y $x $nz $ny $nx $d/pred.log || { log "$B predict failed"; continue; }; fi
  t1=$(date +%s)
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
  rm -rf $d/orc
  # the ceiling of this fitter config (label phase where known, this run's network planes elsewhere); cached per box
  if ! grep -qP "^$B\t" $CEIL; then
    RUN $U fields-oracle $f $PR $PK $d/swap --fill net --phase-scale $PF > $d/swap.log 2>&1 && fitsc $d/swap $d/fitswap $d/fitswap.log
    printf '%s\t%s\t%s\t%s\n' $B $(num $(g_ids $d/fitswap.log)) $(num $(g_pair $d/fitswap.log 1 1)) $FT >> $CEIL; rm -rf $d/swap $d/fitswap
  fi
  read -r cl clp <<< "$(awk -F'\t' -v b=$B '$1 == b {print $2, $3}' $CEIL | tail -1)"
  fr=$(echo "$ids $cl" | awk '$1 != "na" && $2 != "na" && $2 > 0 {printf "%.4f", $1 / $2; exit} {print "na"}')
  echo "$TAG	$B	$sc	$ids	$ids0	$ia	$ib	$sd	$e08$pp	$pe	$gr	$g1	$cf	$cl	$clp	$fr	$((t1 - t0))	$((t2 - t1))	$BIN	$FCFG" >> $RES
  log "$B ids $ids (box $ids0 shifted $ia $ib) p1x $(g_pair $d/fit.log 1 1) gross100 $g1 ceil $cl pred $((t1 - t0))s fit $((t2 - t1))s"
  # (the fit dir is kept: others read runs/TAG/BOX/fit)
  [ -z "$FIELDS" ] && [ "$KEEP" = 0 ] && rm -rf $f
done

# ---- Paris 4 full-cross-section slab sites (in-sample for every Paris 4-trained model) ----
# site s (19.2 um field cells, s = 256 k: the box on the predict window / shard grid): fields cells [s, s+160) x y 512..4096 x 256..4096,
# slabs cropped to the production union crop y 608..3936 x 352..3872 (field cells); slab A = cells
# [0, 96), B = [64, 160) of the site (production 64-cell interiors + 16 overlap), A/B interiors [16, 80) / [80, 144) joined by fit-join.
if [ "$SLABS" = 1 ]; then
  scroll paris4
  [ -s $RS ] || printf 'tag\tbox\tscroll\tids\tids_a\tids_b\tp1x\tp4x\tjoin\tjoined\tjoined_p4x\tfit_s\tbin\tfcfg\n' > $RS
  [ -s $RR ] || printf 'tag\tsite\tslab\tr0_mm\tr1_mm\tvoxels\tm2\tm1\tzero\tp1\tp2\n' > $RR
  for s in $SITES; do
    S=site$s; grep -qP "^$TAG\t$S\t" $RS && { log "$S done"; continue; }
    d=$O/$S; mkdir -p $d; f=$FD/$S; t0=$(date +%s)
    if [ -n "$FIELDS" ]; then [ -e $f/fields.json ] || { log "$S: no fields $f"; continue; }
    elif [ ! -e $f/fields.json ]; then predict $f $((2 * s)) 1024 512 320 7168 7680 $d/pred.log big || { log "$S predict failed"; continue; }; fi
    bash $H/mkcrop.sh $f $d/fA 0 96 96 96 3328 3520 > /dev/null; bash $H/mkcrop.sh $f $d/fB 64 96 96 96 3328 3520 > /dev/null; bash $H/mkcrop.sh $f $d/fC 0 96 96 160 3328 3520 > /dev/null
    for k in A B; do UFSM_FIT_THREADS=$THREADS RUNBIG $U fit $d/f$k --gpu $GPU --out $d/fit$k --codes $QR $QK --score-detail 1 --score-pairs 1 --dir 1 --uw-repair 8 ${FIT//\{box\}/$S} > $d/fit$k.log 2>&1
      grep -E '^    radius' $d/fit$k.log | awk -v t=$TAG -v s=$S -v k=$k '{gsub(/-/, " "); $0 = $0; print t"\t"s"\t"k"\t"$2"\t"$3"\t"$5"\t"$7"\t"$8"\t"$9"\t"$10"\t"$11}' >> $RR; done
    RUN $U fit-join $d/joined $d/fC --blend 16 $d/fitA 32 160 $d/fitB 160 288 > $d/join.log 2>&1
    js=$(num $(grep -oE '^ +0\|1 .*' $d/join.log | awk '{print $NF}'))
    UFSM_FIT_THREADS=$THREADS RUNBIG $U fit $d/fC --model $d/joined --gpu $GPU --codes $QR $QK --score-pairs 1 > $d/joined.log 2>&1
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
log "wbench $TAG finished"

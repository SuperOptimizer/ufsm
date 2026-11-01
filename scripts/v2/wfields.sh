#!/usr/bin/env bash
# wfields.sh CKPT TAG: the winding screen split by the no-fits-on-the-laptop rule (feedback 10-28): the LAPTOP GPU predicts the wgate
# fields (the core boxes + the 4 Paris 4 slab sites, wbench.sh's predict: production window, fit-zarr h 2), they are copied to the
# desk's wbench ($DWB/fields/TAG), and the DESK runs wgate.sh with FIXED=1 (fields as they are, never predicts: CPU fits, the
# pre-registered PASS table vs the desk's wind96 rows). Laptop kit: ~/ufsm-data/wbench (site-laptop.env; a working copy of the desk
# master at /vesuvius/ufsm/wbench, refreshed by /tmp-free kit sync, see the memory note ufsm-v2-train-2026-10-31.md).
# Env: GPU (0), BOXES (core), SITES ("4624 5392 5904 6672"), DESK (forrest@192.168.0.175), DWB (/vesuvius/ufsm/wbench),
#   DREPO (/vesuvius/ufsm/v2/eval/scripts: a copy of scripts/wbench), FITGPU (desk GPU id passed to wgate for its lock, 1), NOFIT=1 (predict only).
set -u
H=$(cd "$(dirname "$0")" && pwd); WBS=$(cd $H/../wbench && pwd)
CK=$(readlink -f "${1:?usage: wfields.sh CKPT TAG}"); TAG=${2:?TAG}
SITE=laptop; GPU=${GPU:-0}; . $WBS/site-laptop.env
U=${U:-$WB/bin/ufsm}; UP=${UP:-$U}; BT=$WBS/boxes.tsv; BOXES=${BOXES:-core}; SITES=${SITES-4624 5392 5904 6672}
DESK=${DESK:-forrest@192.168.0.175}; DWB=${DWB:-/vesuvius/ufsm/wbench}; DREPO=${DREPO:-/vesuvius/ufsm/v2/eval/scripts}
FD=$WB/fields/$TAG; mkdir -p $FD $WB/logs
eval "$(sed -n '/^scroll() {/,/^# ---- boxes ----/p' $WBS/wbench.sh)"   # scroll, predict (the wgate predict recipe)
if [ "$BOXES" = core ]; then BL=$(grep -v '^#' $BT | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1); else BL=$BOXES; fi
log() { echo "$(date '+%F %T') wfields $TAG: $*"; }
scroll paris4
for s in $SITES; do
  f=$FD/site$s; [ -e $f/fields.json ] && continue
  ( flock 9; predict $f $((2 * s)) 1024 512 320 7168 7680 $WB/logs/$TAG-site$s.log big ) 9> $LOCK || log "site$s predict failed ($WB/logs/$TAG-site$s.log)"
done
for B in $BL; do
  IFS=$'\t' read -r nm sc z y x nz ny nx lv note <<< "$(awk -F'\t' -v b=$B '$1 == b' $BT)"; [ -n "${nx:-}" ] || { log "no box $B"; continue; }
  scroll $sc || { log "$B: unknown scroll $sc"; continue; }
  f=$FD/$B; [ -e $f/fields.json ] && continue
  ( flock 9; predict $f $z $y $x $nz $ny $nx $WB/logs/$TAG-$B.log ) 9> $LOCK || log "$B predict failed ($WB/logs/$TAG-$B.log)"
done
log "predicted $(ls -d $FD/*/ 2>/dev/null | wc -l) field sets; copying to $DESK:$DWB/fields/$TAG"
rsync -a $FD/ $DESK:$DWB/fields/$TAG/ || { log "copy failed"; exit 1; }
[ "${NOFIT:-0}" = 1 ] && exit 0
rsync -a $WBS/ $DESK:$DREPO/wbench/ && rsync -a $CK $DESK:$DWB/fields/$TAG.ckpt
log "desk: wgate FIXED=1 (CPU fits) -> $DWB/gate/$TAG/table.txt"
ssh $DESK "cd $DREPO/wbench && SITE=desk FIXED=1 GPU=${FITGPU:-1} BOXES='$BOXES' SITES='$SITES' nohup bash wgate.sh $DWB/fields/$TAG.ckpt $TAG > $DWB/logs-$TAG.wgate.log 2>&1 &"

#!/usr/bin/env bash
# wgate-slabs.sh TAG [CKPT]: the 4 Paris 4 slab sites of a laptop wgate run, computed on forlindesk (60 GB RAM; on the laptop the
# 12-20 GB slab fits get OOM-killed in ufsm.slice). Copies the checkpoint (default: $WB/gate/TAG/ckpt) to forlindesk, runs
# wgate.sh BOXES=none there (site predicts + slab fits, against that site's own wind96 baseline REF=wind96fd: its MX-emulated predict
# differs from the 5080 in the last bits), and copies slabs.tsv / radius.tsv back into $WB/gate/TAG/ with slabref = wind96fd, so
# wgate-table.sh compares them with forlindesk's wind96 slabs. Blocking; queued on the remote GPU's gpuq.
# Env: RHOST (forlindesk), RWB (/vesuvius/ufsm/wgate), RGPU (0), RBIN (remote scripts dir, default $RWB/bin/wb-current),
#   RUP (remote predict binary, $RWB/bin/ufsm), RREF (wind96fd).
set -u
H=$(cd "$(dirname "$0")" && pwd)
SITE=${SITE:-laptop}; GPU=0; . $H/site-$SITE.env
TAG=$1; O=$WB/gate/$TAG; CK=${2:-$(cat $O/ckpt 2> /dev/null)}
RHOST=${RHOST:-forlindesk}; RWB=${RWB:-/vesuvius/ufsm/wgate}; RGPU=${RGPU:-0}; RBIN=${RBIN:-$RWB/bin/wb-current}; RUP=${RUP:-$RWB/bin/ufsm}; RREF=${RREF:-wind96fd}
[ -s "$CK" ] || { echo "wgate-slabs: no checkpoint for $TAG"; exit 2; }
mkdir -p $O
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $O/run.log; }
S="ssh -o BatchMode=yes $RHOST"
log "slab sites of $TAG on $RHOST GPU $RGPU (vs $RREF)"
$S "mkdir -p $RWB/ckpt" && rsync -a "$CK" $RHOST:$RWB/ckpt/$TAG.ckpt || { log "wgate-slabs: copy to $RHOST failed"; exit 1; }
$S "cd $RWB && SITE=forlindesk GPU=$RGPU UP=$RUP REF=$RREF BOXES=none SITES='${SITES:-4624 5392 5904 6672}' TTA=${TTA:-1} SEAM=${SEAM:-0} PREV=${PREV:-} bash $RBIN/wgate.sh $RWB/ckpt/$TAG.ckpt $TAG > $RWB/gate-$TAG-slabs.out 2>&1"
for f in slabs.tsv radius.tsv; do rsync -a $RHOST:$RWB/gate/$TAG/$f $O/$f.remote || { log "wgate-slabs: no $f from $RHOST"; exit 1; }; done
# keep the laptop's own slab rows (if any) aside; the remote rows become the run's slab rows
for f in slabs radius; do [ -s $O/$f.tsv ] && [ ! -e $O/$f.local.tsv ] && mv $O/$f.tsv $O/$f.local.tsv; mv $O/$f.tsv.remote $O/$f.tsv; done
echo $RREF > $O/slabref
for f in slabs radius; do rsync -a $RHOST:$RWB/gate/$RREF/$f.tsv $WB/res/$RREF-$f.tsv 2> /dev/null; done   # the reference rows
log "slab sites of $TAG: $(awk -F'\t' 'NR > 1 {printf "%s %s  ", $2, $10}' $O/slabs.tsv)"

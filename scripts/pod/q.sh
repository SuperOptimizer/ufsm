#!/bin/bash
# q.sh LANE: whole-scroll latent unrolls (u1-13750, RECTOSP=0), smallest first (kit/q.txt, user 11-12); 2 lanes share the card
# (FITLANES 2, TH 13 each). A lane claims the next scroll (flock), fetches its CT (setup.sh step 6), uses the corpus box / axis, runs pod/unroll.sh.
# Stop after the current scroll: touch /root/pod/STOPQ
# 11-13 (pod disk full at 00:24): a lane waits while / has < MINDISK GB free; a claim counts only when its file was written; a failed
# setup marks the scroll failed and the lane backs off 60 s (no tight loop); a run that ends rc 0 without state/stripimg.done is a failure.
LN=${1:?}; POD=/root/pod; Q=$POD/kit/q.txt; QD=$POD/qstate; mkdir -p $QD; MINDISK=${MINDISK:-60}
log() { echo "$(date "+%F %T") lane $LN: $*" | tee -a $POD/q.log; }
freegb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
while :; do
  [ -e $POD/STOPQ ] && { log "STOPQ: lane stops"; exit 0; }
  w=0; while [ "$(freegb)" -lt $MINDISK ]; do [ $w = 0 ] && log "disk: $(freegb) GB free < $MINDISK: waiting"; w=1; sleep 60; done
  S=$(flock $QD/.lock bash -c "for s in \$(cat $Q); do [ -e $QD/\$s.claim ] && continue; touch $QD/\$s.claim && echo \$s; break; done")
  [ -z "$S" ] && { if [ -n "$(flock $QD/.lock bash -c "for s in \$(cat $Q); do [ -e $QD/\$s.claim ] || echo \$s; done")" ]; then log "claim failed: retry in 60 s"; sleep 60; continue; fi; log "queue empty"; exit 0; }
  mkdir -p $POD/scrolls/$S; cp $POD/kit/$S/box.txt $POD/kit/$S/axis.json $POD/scrolls/$S/ 2>/dev/null
  log "$S fetch"; t0=$(date +%s)
  if ! POD=$POD NOAPT=1 NOBUILD=1 NOSMOKE=1 MTH=12 bash $POD/ufsm/scripts/pod/setup.sh $S > $POD/setup-$S.out 2>&1; then
    log "$S setup FAILED ($POD/setup-$S.out; $(freegb) GB free): marked failed, next in 60 s"; touch $QD/$S.fail; sleep 60; continue; fi
  cp $POD/kit/$S/box.txt $POD/scrolls/$S/box.txt; B=$(cat $POD/kit/$S/box.txt)
  sed -i "s#CK=$POD/ckpt/wind96-wk1-best.ckpt#CK=$POD/ckpt/u1-13750.ckpt#; s#BOX=[0-9,]*;#BOX=$B;#" $POD/scrolls/$S.env
  [ -e $POD/unroll/$S/axis.json ] || sed -i "s#AXS=[^;]*;#AXS=$POD/scrolls/$S/axis.json;#" $POD/scrolls/$S.env   # (a resumed run keeps its axis)
  log "$S unroll start (fetch $(( $(date +%s) - t0 )) s; $(grep -o "BOX=[^;]*" $POD/scrolls/$S.env))"
  cd $POD/ufsm && RECTOSP=0 FITLANES=2 TH=13 PTH=13 ITH=13 UFSM_NCPU=13 bash scripts/pod/unroll.sh $S > $POD/unroll-$S.out 2>&1; rc=$?
  [ $rc = 0 ] && [ ! -e $POD/unroll/$S/state/stripimg.done ] && rc=9
  log "$S unroll rc=$rc ($(( $(date +%s) - t0 )) s)"; [ $rc = 0 ] && touch $QD/$S.ok || touch $QD/$S.fail
done

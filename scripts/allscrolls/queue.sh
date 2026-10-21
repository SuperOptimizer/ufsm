#!/usr/bin/env bash
# queue.sh table|desk-fit|desk-ink|desk-render|laptop|status: the all-scrolls job driver (common.sh has the layout). Restart-safe:
# each step leaves A/<S>/state/<step>.done (a failed step <step>.failed, retried while fewer than 3 failures).
#   table        A/scrolls.tsv from the mirror listing (2025/2026 ESRF 8.64 / 9.36 um scans with prod96 fit fields)
#   desk-fit     local 2 cm box fits (fit.sh), in passes of MAXBOX boxes per scroll (PASSES "2 6 24": the best boxes first, every
#                scroll before widening); scrolls whose ink is done first. Run 2 lanes (they share a scroll's boxes by claims).
#                CPU fits (FITCPU=1) until desk-ink is done (A/ink.alldone), then the emptier GPU.
#   desk-ink     ink.sh per scroll once no other process holds a GPU (scrolls with surfaces first), both GPUs
#   desk-render  render.sh for every scroll with ink and new surfaces (incremental), every minute
#   laptop       local 1 cm box fits (fit.sh, N 512, 6 GB cap) from the END of the scroll order (passes "2 6"), surfaces pushed to the desk
#                (A/<S>/surf-l); pulls renders/, sheets/, sheets.tsv, fits tables to ~/ufsm-data/allscrolls/<S>; STATUS.txt
#   status       one line per scroll (on the laptop it asks the desk)
# Start: setsid nohup scripts/allscrolls/queue.sh LANE > A/lane-LANE.log 2>&1 < /dev/null &   Stop: touch A/STOP (between steps).
D=$(cd "$(dirname "$0")" && pwd); . $D/common.sh
cmd=${1:-status}
order() { for s in $FIRST; do grep -q "^$s	" $A/scrolls.tsv && echo $s; done; awk '{print $1}' $A/scrolls.tsv | while read -r s; do [[ " $FIRST " == *" $s "* ]] || echo $s; done; }
nfail() { cat $A/$1/state/$2.failed 2> /dev/null | wc -l; }
fail() { mkdir -p $A/$1/state; echo "$(date '+%F %T') ${3:-}" >> $A/$1/state/$2.failed; log "$1 $2 FAILED ${3:-}"; }
stop() { [ -e $A/STOP ]; }
desk_only() { [ $HOST = desk ] || { echo "queue.sh $cmd: run it on the desk"; exit 2; }; }
case $cmd in
table)
  mkdir -p $A; t=$A/scrolls.tsv.tmp; : > $t
  for s in $(curl -sf $R/ | grep -oE 'href="PHerc[^"/]+/"' | sed 's/href="//; s#/"##'); do
    v=$(curl -sf $R/$s/volumes/ | grep -oE 'href="202[56][^"?]+-(8\.640|9\.362|9\.366)um[^"?]*-masked\.zarr/"' | head -1 | sed 's/href="//; s#/"##')
    [ -n "$v" ] || continue
    f=$(curl -sf $R/$s/representations/predictions/fit-fields/ | grep -oE "href=\"${v%%-*}-fitfields-[^\"?]+prod96-h2\.zarr/\"" | head -1 | sed 's/href="//; s#/"##')
    [ -n "$f" ] || { echo "$s: $v has no prod96 fit fields, skipped" >&2; continue; }
    um=$(echo $v | grep -oE '[0-9]+\.[0-9]+um' | tr -d um)
    printf '%s\t%s\t%s\t%s\n' $s ${v%.zarr} $um $f >> $t
  done
  mv $t $A/scrolls.tsv; wc -l < $A/scrolls.tsv;;
desk-fit)
  desk_only
  for mb in ${PASSES:-2 6 24}; do
    while ! stop; do
      left=0; did=0
      for s in $( { for s in $(order); do done_ $s ink && echo $s; done; for s in $(order); do done_ $s ink || echo $s; done; } ); do
        stop && break; done_ $s fit-desk-1024-$mb && continue; [ $(nfail $s fit-$mb) -ge 3 ] && continue; left=1
        mkdir -p $A/$s; [ -e $A/ink.alldone ] && unset FITCPU || export FITCPU=1   # the GPUs belong to the ink lane until it is done
        REPICK=1 MAXBOX=$mb $D/fit.sh $s >> $A/$s/fit-lane$$.log 2>&1; rc=$?
        case $rc in 0) log "$s fit pass $mb done ($(cat $A/$s/state/fit-desk-1024-$mb.done))"; did=1;; 4) ;; *) fail $s fit-$mb "rc $rc"; did=1;; esac
      done
      [ $left = 0 ] && break; [ $did = 0 ] && sleep 120
    done
  done; log "desk-fit: passes ${PASSES:-2 6 24} done";;
desk-ink)
  desk_only
  while ! stop; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; then sleep 300; continue; fi   # (none of them ours)
    next=""; for s in $(order); do { done_ $s ink || [ $(nfail $s ink) -ge 3 ]; } && continue; [ -z "$next" ] && next=$s; [ -s $A/$s/surf/list.tsv ] && { next=$s; break; }; done
    [ -n "$next" ] || { log "desk: all ink done"; date > $A/ink.alldone; break; }
    log "$next ink start"; mkdir -p $A/$next; $D/ink.sh $next >> $A/$next/ink-lane.log 2>&1; rc=$?
    case $rc in 0) log "$next ink done ($(cat $A/$next/state/ink.done))";; 3) log "desk-ink: /vesuvius is nearly full, stopping"; break;; *) fail $next ink "rc $rc";; esac
  done;;
desk-render)
  desk_only
  while ! stop; do
    for s in $(order); do
      stop && break; done_ $s ink || continue
      new=0; for L in surf surf-l; do [ -s $A/$s/$L/list.tsv ] && [ $A/$s/$L/list.tsv -nt $A/$s/renders/index.tsv ] && new=1; done
      [ $new = 1 ] && [ $(nfail $s render) -lt 3 ] || continue
      if $D/render.sh $s >> $A/$s/render-lane.log 2>&1; then log "$s rendered: $(cat $A/$s/state/render.done)"; else fail $s render "rc $?"; fi
    done
    sleep 60
  done;;
laptop)
  [ $HOST = laptop ] || { echo "queue.sh laptop: run it on the laptop"; exit 2; }
  pull() { for s in $(order); do ssh $DESK "test -s $DA/$s/sheets.tsv" 2> /dev/null || continue; mkdir -p $A/$s
      rsync -a $DESK:$DA/$s/renders $DESK:$DA/$s/sheets $DESK:$DA/$s/sheets.tsv $A/$s/ 2> /dev/null; rsync -a $DESK:$DA/$s/fits-desk.tsv $A/$s/ 2> /dev/null; done
    ssh $DESK "$DA/scripts/queue.sh status" > $A/STATUS.txt 2> /dev/null; }
  for mb in ${PASSES:-2 6}; do
    for s in $(order | tac); do
      stop && break; pull; done_ $s fit-laptop-512-$mb && continue; [ $(nfail $s fitl-$mb) -ge 3 ] && continue
      MAXBOX=$mb $D/fit.sh $s >> $A/$s.fit-laptop.log 2>&1; rc=$?
      [ $rc = 0 ] || [ $rc = 4 ] || fail $s fitl-$mb "rc $rc"
      [ -s $A/$s/surf-l/list.tsv ] && ssh $DESK "mkdir -p $DA/$s" && rsync -a $A/$s/surf-l $DESK:$DA/$s/ && log "$s laptop surfaces pushed"
      rm -rf $A/$s/fields $A/$s/state/fields.done   # (refetched for the next pass)
    done
  done
  while ! stop; do pull; sleep 600; done;;
status)
  [ $HOST = laptop ] && { ssh $DESK "$DA/scripts/queue.sh status"; exit; }
  printf '%-12s %6s %6s %6s  %-5s %8s  %s\n' scroll um boxes boxes1 ink rendered "largest rendered sheet (cm2, on papyrus)"
  for s in $(order); do um=$(awk -v s=$s '$1 == s {print $3}' $A/scrolls.tsv)
    nb=$(cat $A/$s/fits-desk.tsv 2> /dev/null | wc -l); nl=$(ls -d $A/$s/boxes-512/*/FITDONE 2> /dev/null | wc -l)
    nr=$(( $(cat $A/$s/renders/index.tsv 2> /dev/null | wc -l) - 1 )); [ $nr -lt 0 ] && nr=0
    big=$(tail -n +2 $A/$s/renders/index.tsv 2> /dev/null | sort -t$'\t' -k3,3 -gr | head -1 | awk -F'\t' '{print $1, $3, $4}')
    printf '%-12s %6s %6s %6s  %-5s %8s  %s\n' $s $um $nb $nl $(done_ $s ink && echo done || echo -) $nr "${big:--}"
  done;;
*) echo "usage: queue.sh table|desk-fit|desk-ink|desk-render|laptop|status"; exit 2;;
esac

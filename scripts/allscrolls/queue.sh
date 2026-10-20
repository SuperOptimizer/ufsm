#!/usr/bin/env bash
# queue.sh table|desk-fit|desk-ink|desk-render|laptop|status: the all-scrolls job driver (common.sh has the layout). Every loop is
# resumable: each step leaves A/<S>/state/<step>.done, a failed step <step>.failed (retried while fewer than 3 failures).
#   table        A/scrolls.tsv from the mirror listing (2025/2026 ESRF 8.64 / 9.36 um scans with prod96 fit fields)
#   desk-fit     fit.sh per scroll in order (MAXBOX boxes per scroll, default 24: the middle-z rows); run 2 lanes: they share the boxes
#                of a scroll (claims) and move on when the rest is claimed. CPU fits (FITCPU=1) while desk-ink holds the GPUs.
#   desk-ink     ink.sh per scroll once no other process holds a GPU (scrolls with surfaces first), both GPUs
#   desk-render  render.sh per scroll with surfaces and ink
#   laptop       pulls the renders, index pages and fit tables to the laptop (A/<S>/renders, fits.tsv, gtcheck.txt); STATUS.txt
#   status       one line per scroll (run on either host; on the laptop it asks the desk)
# The laptop's 6 GB-per-job cap OOMs 1-2 cm box fits (ct-fields, fit: 6.3 GB at 768^3 cells), so the fits run on the desk.
# Start (restart-safe): setsid nohup scripts/allscrolls/queue.sh LANE > A/LANE.log 2>&1 < /dev/null &   Stop: touch A/STOP.
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
  while ! stop; do
    did=0
    for s in $(order); do
      stop && break
      { done_ $s fit || [ $(nfail $s fit) -ge 3 ]; } && continue
      mkdir -p $A/$s; [ -e $A/ink.alldone ] || export FITCPU=${FITCPU:-1}   # the GPUs belong to the ink lane until it is done
      MAXBOX=${MAXBOX:-24} $D/fit.sh $s >> $A/$s/fit-lane$$.log 2>&1; rc=$?
      case $rc in 0) log "$s fit done ($(cat $A/$s/state/fit.done))"; did=1;; 4) ;; *) fail $s fit "rc $rc"; did=1;; esac
    done
    [ $did = 0 ] && sleep 300
  done;;
desk-ink)
  desk_only
  while ! stop; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" != 0 ]; then sleep 300; continue; fi   # (none of them ours)
    next=""; for s in $(order); do { done_ $s ink || [ $(nfail $s ink) -ge 3 ]; } && continue; [ -z "$next" ] && next=$s; done_ $s surf && { next=$s; break; }; done
    [ -n "$next" ] || { log "desk: all ink done"; date > $A/ink.alldone; break; }
    log "$next ink start"; mkdir -p $A/$next; $D/ink.sh $next >> $A/$next/ink-lane.log 2>&1; rc=$?
    case $rc in 0) log "$next ink done ($(cat $A/$next/state/ink.done))";; 3) log "desk-ink: /vesuvius is nearly full, stopping"; break;; *) fail $next ink "rc $rc";; esac
  done;;
desk-render)
  desk_only
  while ! stop; do
    did=0
    for s in $(order); do
      stop && break
      done_ $s surf && done_ $s ink && ! done_ $s render && [ $(nfail $s render) -lt 3 ] || continue
      log "$s render start"; if $D/render.sh $s >> $A/$s/render-lane.log 2>&1; then log "$s render done"; else fail $s render "rc $?"; fi; did=1
    done
    [ $did = 0 ] && sleep 600
  done;;
laptop)
  [ $HOST = laptop ] || { echo "queue.sh laptop: run it on the laptop"; exit 2; }
  while ! stop; do
    for s in $(order); do
      ssh $DESK "test -e $DA/$s/state/render.done" 2> /dev/null && ! done_ $s pull || continue
      mkdir -p $A/$s && rsync -a $DESK:$DA/$s/renders $DESK:$DA/$s/fits.tsv $DESK:$DA/$s/surf/list.tsv $A/$s/ && { rsync -a $DESK:$DA/$s/gtcheck.txt $A/$s/ 2> /dev/null; true; } &&
        mark $s pull && log "$s renders pulled to $A/$s/renders"
    done
    ssh $DESK "$DA/scripts/queue.sh status" > $A/STATUS.txt 2> /dev/null
    sleep 900
  done;;
status)
  [ $HOST = laptop ] && { ssh $DESK "$DA/scripts/queue.sh status"; exit; }
  st() { done_ $s $1 && echo done || { [ -e $A/$s/state/$1.failed ] && echo failed || echo -; }; }
  printf '%-12s %6s %5s  %-18s %-6s %-6s  %s\n' scroll um boxes fit ink render "largest surface (cm2)"
  for s in $(order); do um=$(awk -v s=$s '$1 == s {print $3}' $A/scrolls.tsv)
    nb=$(cat $A/$s/fits.tsv 2> /dev/null | wc -l); ng=$(awk '$3 == "global"' $A/$s/fits.tsv 2> /dev/null | wc -l)
    big=$(head -1 $A/$s/surf/list.tsv 2> /dev/null | cut -f1,3 | tr '\t' ' ')
    printf '%-12s %6s %5s  %-18s %-6s %-6s  %s\n' $s $um $nb "$(st fit) ($ng global)" $(st ink) $(st render) "${big:--}"
  done;;
*) echo "usage: queue.sh table|desk-fit|desk-ink|desk-render|laptop|status"; exit 2;;
esac

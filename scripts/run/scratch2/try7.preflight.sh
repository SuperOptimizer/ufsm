#!/bin/bash
# try7.preflight.sh (merger2 11-14): the checks BEFORE `s2ctl.sh fresh try7.launch.env ...` (audit-data's preconditions + mine). rc 0 = go.
# After the fresh dry start, run it again with --after to check the dry log (s2ctl.log.dry-holdouts / .dry-last.log).
set -u
R=/vesuvius/ufsm/v2/lat/runs; E=$R/try7.launch.env; ok=1; bad() { echo "FAIL: $*"; ok=0; }; good() { echo "ok: $*"; }
U=$(grep '^U=' $E | cut -d= -f2-); SRCJ=$(grep '^SRCJ=' $E | cut -d= -f2-)
if [ "${1:-}" != --after ]; then
  [ -x "$U" ] && good "binary $U" || bad "binary not executable: $U"
  SHA=${U##*.}; if git -C /home/forrest/ufsm cat-file -e "$SHA^{commit}" 2>/dev/null; then
    git -C /home/forrest/ufsm merge-base --is-ancestor cf4dcc8 $SHA 2>/dev/null && good "binary sha $SHA contains datafix cf4dcc8" || bad "binary sha $SHA lacks datafix cf4dcc8"
    git -C /home/forrest/ufsm merge-base --is-ancestor $SHA origin/master 2>/dev/null && good "binary sha $SHA is on origin/master" || bad "binary sha $SHA is not on origin/master"
  else bad "cannot resolve the binary's sha '$SHA' (name it ufsm.scratch2.<sha>)"; fi
  strings "$U" | grep -qF -- "--holdout-guard" && good "binary knows --holdout-guard" || bad "binary lacks --holdout-guard"
  [ -f "$SRCJ" ] && good "sources $SRCJ" || bad "no sources $SRCJ"
  python3 -I -c "import json,sys; d=json.load(open('$SRCJ')); s=d['sources']; w=[x for x in s if x['name']=='PHerc1218-b3']; sys.exit(0 if d.get('holdout_strict')==1 and all(x.get('holdout_file') for x in s) and w and w[0]['weight']==0 else 1)" \
    && good "sources: holdout_strict 1, holdout_file on every source, PHerc1218-b3 weight 0" || bad "sources json lacks holdout_strict / holdout_file / 1218 weight 0"
  [ -f /vesuvius/ufsm/univ/scratch1/holdouts-20261114.json ] && good "holdouts json" || bad "no holdouts-20261114.json"
  for r in p4-codesA48 p4-codesB48; do [ -f /vesuvius/ufsm/datafix-20261114/raster/$r/done ] && good "raster $r done" || bad "raster $r not done"; done
  grep -q ALLDONE /vesuvius/ufsm/datafix-20261114/raster/run.log 2>/dev/null && good "raster run.log ALLDONE" || bad "raster run.log has no ALLDONE"
  [ -f $R/scratch2.parked ] && good "try6 parked (fresh clears it)"
  systemctl --user is-active ufsm-health@scratch2 > /dev/null 2>&1 && good "health service active" || bad "ufsm-health@scratch2 not active"
else
  H=$R/s2ctl.log.dry-holdouts; F=$R/s2ctl.log.dry-last.log; [ -f $F ] || { echo "FAIL: no dry log $F"; exit 1; }
  n=$(python3 -I -c "import json; print(sum(1 for x in json.load(open('$SRCJ'))['sources'] if x['weight'] > 0))")
  h=$(grep -cE '[1-9][0-9]* held-out boxes from .*holdouts-20261114.json' $F); z=$(grep -cE '(^|[^0-9])0 held-out boxes from' $F)
  [ "$h" -ge "$n" ] && [ "$z" = 0 ] && good "$h sources with held-out boxes (>= $n weighted sources), none with 0" || bad "held-out lines: $h with boxes, $z with 0, $n weighted sources"
  grep -q "lies outside the CT" $F && bad "'lies outside the CT' in the dry log" || good "no box outside the CT"
  grep -qE "val by (source|scroll)" $F && good "val by source / scroll lines" || bad "no 'val by source / scroll' line in the dry log"
fi
[ $ok = 1 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED"; [ $ok = 1 ]

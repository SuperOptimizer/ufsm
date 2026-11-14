#!/bin/bash
# try7.preflight.sh (merger2 11-14): the checks BEFORE `s2ctl.sh fresh try7.launch.env ...` (audit-data's preconditions + mine). rc 0 = go.
# After the fresh dry start, run it again with --after to check the dry log (s2ctl.log.dry-holdouts / .dry-last.log).
set -u
R=/vesuvius/ufsm/v2/lat/runs; E=$R/try7.launch.env; ok=1; bad() { echo "FAIL: $*"; ok=0; }; good() { echo "ok: $*"; }
U=$(grep '^U=' $E | cut -d= -f2-); SRCJ=$(grep '^SRCJ=' $E | cut -d= -f2-)
if [ "${1:-}" != --after ]; then
  [ -x "$U" ] && good "binary $U" || bad "binary not executable: $U"
  SHA=${U##*.}; if git -C /home/forrest/ufsm cat-file -e "$SHA^{commit}" 2>/dev/null; then
    git -C /home/forrest/ufsm merge-base --is-ancestor 9a3ea4a $SHA 2>/dev/null && good "binary sha $SHA contains datafix 9a3ea4a (UFMC0002 packs, cf4dcc8)" || bad "binary sha $SHA lacks datafix 9a3ea4a"
    git -C /home/forrest/ufsm merge-base --is-ancestor $SHA origin/master 2>/dev/null && good "binary sha $SHA is on origin/master" || bad "binary sha $SHA is not on origin/master"
  else bad "cannot resolve the binary's sha '$SHA' (name it ufsm.scratch2.<sha>)"; fi
    git -C /home/forrest/ufsm merge-base --is-ancestor 472df19 $SHA 2>/dev/null && good "binary contains sampbias 472df19 (zboost: sources.c ignores unknown keys silently)" || bad "binary lacks sampbias 472df19"
    for b in dcachecap-20261114 dckey-20261114; do git -C /home/forrest/ufsm merge-base --is-ancestor origin/$b $SHA 2>/dev/null && good "binary contains $b" || bad "binary lacks origin/$b (hostmem: derived-cache cap + new keys)"; done
  [ -e /home/forrest/.cache/ufsm/derived ] && bad "~/.cache/ufsm/derived still in place: mv it to derived.try6 (no trainer / admuon arm reading it)" || good "old derived cache moved away"
  strings "$U" | grep -qF -- "--holdout-guard" && good "binary knows --holdout-guard" || bad "binary lacks --holdout-guard"
  [ -f "$SRCJ" ] && good "sources $SRCJ" || bad "no sources $SRCJ"
  python3 -I -c "import json,sys; d=json.load(open('$SRCJ')); s=d['sources']; w=[x for x in s if x['name']=='PHerc1218-b3']; sys.exit(0 if d.get('holdout_strict')==1 and all(x.get('holdout_file') for x in s) and w and w[0]['weight']==0 else 1)" \
    && good "sources: holdout_strict 1, holdout_file on every source, PHerc1218-b3 weight 0" || bad "sources json lacks holdout_strict / holdout_file / 1218 weight 0"
  [ -f /vesuvius/ufsm/univ/scratch1/holdouts-20261114.json ] && good "holdouts json" || bad "no holdouts-20261114.json"
  for r in p4-codesA48 p4-codesB48; do [ -f /vesuvius/ufsm/datafix-20261114/raster/$r/done ] && good "raster $r done" || bad "raster $r not done"; done
  grep -q ALLDONE /vesuvius/ufsm/datafix-20261114/raster/run.log 2>/dev/null && good "raster run.log ALLDONE" || bad "raster run.log has no ALLDONE"
  if [ -f $R/scratch2.audit.ok ] && [ $R/scratch2.audit.ok -nt $E ] && [ $R/scratch2.audit.ok -nt "$U" ]; then good "leak audit: $(cat $R/scratch2.audit.ok)"
  else bad "no leak audit newer than the env and binary: S2CTL_OWNER=merger2 bash $R/s2ctl.sh audit $E 1000"; fi
  [ -f $R/scratch2.parked ] && good "try6 parked (fresh clears it)"
  systemctl --user is-active ufsm-health@scratch2 > /dev/null 2>&1 && good "health service active" || bad "ufsm-health@scratch2 not active"
else
  H=$R/s2ctl.log.dry-holdouts; F=$R/s2ctl.log.dry-last.log; [ -f $F ] || { echo "FAIL: no dry log $F"; exit 1; }
  # (audit-data's format: "sources: <name>: <N> held-out boxes from <path> (...)"; every source prints it, weight 0 included)
  n=$(python3 -I -c "import json; print(len(json.load(open('$SRCJ'))['sources']))")
  h=$(grep -E '^sources: .*: [1-9][0-9]* held-out boxes from .*holdouts-20261114.json' $F | sed -E 's/^sources: (.*): [0-9]+ held-out boxes.*/\1/' | sort -u | wc -l)
  z=$(grep -cE '^sources: .*: 0 held-out boxes from' $F)
  [ "$h" -ge "$n" ] && [ "$z" = 0 ] && good "$h of $n sources with held-out boxes, none with 0" || bad "held-out lines: $h of $n sources with boxes, $z with 0"
  grep -qE "^sources: .*: held-out box .* lies outside the CT" $F && bad "'lies outside the CT' in the dry log" || good "no box outside the CT"
  grep -qE "^val by source \(" $F && grep -qE "^val by scroll \(" $F && good "val by source / scroll lines" || bad "no 'val by source / scroll' line in the dry log"
fi
[ $ok = 1 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED"; [ $ok = 1 ]

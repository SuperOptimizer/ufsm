#!/bin/bash
# merger2 11-14 (ops audit #4 / #5, eval audit #6): gated binary swap through s2ctl.sh. BINARY ONLY: sources / settings never change in the
# same swap (eval #6 (2)); change launch.env in its own s2ctl resume. The log gate (eval #6 (3)): medians of NL (12) full-interval step lines
# of ONE trainer process each: the old process's last NL (after its "resumed ... at step" line, its first line skipped as partial) vs
# the new process's first NL (its first line skipped). Roll back if the acc median drops > 0.05 or the gn median doubles. Measured noise
# of 12-line medians: sd ~0.02, so the gate is ~2.5 sigma; it only catches gross breaks. The proper test is a fixed --det replay at lr 0
# on frozen windows, old vs new binary, before the swap (eval #6 (1); u1x.sh style) -- to be added.
# usage: gswap.sh SHA
set -u
SHA=${1:?sha}; [ $# = 1 ] || { echo "gswap: binary only (eval audit #6): change launch.env in a separate s2ctl resume"; exit 2; }
R=/vesuvius/ufsm/v2/lat/runs; o=$R/scratch2; E=$o/launch.env; L=$o.log; NB=/vesuvius/ufsm/v2/lat/bin/ufsm.scratch2.$SHA
NL=${NL:-12}; B=/vesuvius/ufsm/merge/swap2/gate.$SHA; export S2CTL_OWNER=merger2
[ -x $NB ] || { echo "no $NB"; exit 1; }
case "$B" in /vesuvius/ufsm/merge/swap2/gate.*) rm -rf -- "$B";; esac; mkdir -p $B
proc_lines() { awk '/^resumed .* at step /{n++; delete a; k=0; next} n && /^step +[0-9]+ /{a[++k]=$0} END{for(i=2;i<=k;i++) print a[i]}' $L; }   # the current process's lines, first skipped
med() { awk -v f="$1" '{ if (match($0, " " f " [0-9.]+")) { split(substr($0,RSTART,RLENGTH),x," "); v[++n]=x[2] } }
  END { if (!n) {print -1; exit} asort(v); print (n%2 ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2) }'; }
proc_lines | tail -$NL > $B/before.lines; N0=$(wc -l < $B/before.lines)
[ "$N0" -ge $NL ] || { echo "gswap: the current process has only $N0 full lines (< $NL): wait"; exit 1; }
A0=$(med acc < $B/before.lines); G0=$(med gn < $B/before.lines); echo "== baseline (current process, last $NL lines): acc median $A0, gn median $G0"
cp $E $B/launch.env.before
bash $R/s2ctl.sh stop "gswap $SHA: save the previous checkpoint" || exit 1
cp $o/last.ckpt $B/last.ckpt
bash $R/s2ctl.sh resume $NB "gswap $SHA" || { echo "== resume failed: restoring"; cp $B/launch.env.before $E; bash $R/s2ctl.sh resume "" "gswap $SHA failed to start"; exit 1; }
until [ "$(proc_lines | wc -l)" -ge $NL ]; do
  kill -0 "$(cat $o.pid 2>/dev/null)" 2>/dev/null || { sleep 90; kill -0 "$(cat $o.pid 2>/dev/null)" 2>/dev/null || { echo "== TRAINER GONE $(date +%T)"; break; }; }; sleep 60; done
proc_lines | head -$NL > $B/after.lines; N1=$(wc -l < $B/after.lines); A1=$(med acc < $B/after.lines); G1=$(med gn < $B/after.lines)
echo "== after the swap, first $NL lines ($N1): acc median $A1, gn median $G1 vs $A0 / $G0"
if kill -0 "$(cat $o.pid 2>/dev/null)" 2>/dev/null && [ "$N1" -ge $NL ] && python3 -c "import sys; sys.exit(0 if ($A0 - $A1) <= 0.05 and $G1 < 2 * $G0 else 1)"; then
  echo "== GATE PASS $(date +%T)"; exit 0; fi
echo "== GATE FAIL $(date +%T): rolling back"
bash $R/s2ctl.sh stop "gswap $SHA gate fail: acc $A1 vs $A0, gn $G1 vs $G0" || exit 1
mv $o/last.ckpt $o/last.ckpt.gatefail.$SHA; cp $B/last.ckpt $o/last.ckpt; cp $B/launch.env.before $E
bash $R/s2ctl.sh resume "" "gswap $SHA rollback"; echo "== rolled back $(date +%T) to $(grep ^U= $E)"

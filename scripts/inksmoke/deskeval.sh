#!/usr/bin/env bash
# scripts/inksmoke/deskeval.sh PULLDIR: the full ink retrain's held-out score on the desk, chained on the pod's pushed snapshots
# (podq.sh FULL=1 copies each arm's snapshots into K/out as inksmoke{A,B}-full-STEP.ckpt; podq2.sh pushes K/out to PULLDIR).
# Waits for both arms' final snapshots (STEPS 3000), scores each with /vesuvius/ufsm/inksmoke/score.sh (latink over the 1667 box at
# level 2 + the 17 labelled wraps rendered and gated: ~13 min of desk card 1 under the shared locks), then pairwrap.py on the 1667
# held-out split, paired against reader3 (0.8583; inkpod 0.5824) -> /vesuvius/ufsm/inksmoke/eval/pair_full.tsv; then the earlier
# snapshots (1000, 2000) and the table again with them. Writes /vesuvius/ufsm/inksdk/full_eval.done at the end.
# Run: setsid nohup bash deskeval.sh PULLDIR > /vesuvius/ufsm/inksmoke/eval/deskeval.out 2>&1 < /dev/null &
set -uo pipefail; PD=${1:?PULLDIR}; STEPS=${STEPS:-3000}; E=/vesuvius/ufsm/inksmoke/eval; L=$E/deskeval.log
SC=/vesuvius/ufsm/inksmoke/score.sh; I=/vesuvius/ufsm/podun/PHerc1667/ink; DONE=/vesuvius/ufsm/inksdk/full_eval.done
say() { echo "$(date '+%F %T') deskeval: $*" | tee -a $L; }
ck() { find "$PD" -name "inksmoke$1-full-$2.ckpt" -size +1M 2>/dev/null | head -1; }
table() {   # table OUT TAG...: pairwrap over reader3, inkpod and the scored snapshots
  local out=$1; shift; local m="reader3=$I/reader3/map,inkpod=$E/inkpod15000/map"; for t in "$@"; do m="$m,$t=$E/$t/map"; done
  ( cd /vesuvius/ufsm/argus && env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ~/bin/lowprio ~/bin/memcap -m 38G python3 -I pairwrap.py \
      /vesuvius/ufsm/podun/PHerc1667/inksdk/up /vesuvius/ufsm/inkgt/segs-PHerc1667.json $out --ct-key PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr \
      --maps $m --ref reader3 --nonzero reader3 --split heldout > ${out%.tsv}.log 2>&1 ) || { say "pairwrap failed (${out%.tsv}.log)"; return 1; }
  say "table $out:"; column -t $out | tee -a $L; }
score() { local c; c=$(ck $1 $2); [ -n "$c" ] || return 1; mkdir -p $E/ckpt; cp -n $c $E/ckpt/; say "score $1$2 ($c)"
  bash $SC $1$2 $E/ckpt/$(basename $c) >> $L 2>&1 || { say "score $1$2 failed"; return 2; }; }
say "waiting for inksmokeA-full-$STEPS.ckpt and inksmokeB-full-$STEPS.ckpt under $PD"
until [ -n "$(ck A $STEPS)" ] && [ -n "$(ck B $STEPS)" ]; do
  [ -n "$(ck A $STEPS)" ] && [ ! -s $E/A$STEPS/map/w051.tif ] && score A $STEPS   # arm A while arm B trains
  sleep 120; done
sleep 60   # (the push of the last file settles)
score A $STEPS; score B $STEPS
table $E/pair_full.tsv A$STEPS B$STEPS
done_tags=(A$STEPS B$STEPS)
for a in A B; do for s in 1000 2000; do score $a $s && done_tags+=($a$s); done; done
[ ${#done_tags[@]} -gt 2 ] && table $E/pair_full_all.tsv "${done_tags[@]}"
echo "$(date '+%F %T') $E/pair_full.tsv" > $DONE; say "done"

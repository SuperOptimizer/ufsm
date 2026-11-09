#!/usr/bin/env bash
# scripts/rv/good.sh [SCROLLS]: the GOOD PARTS of our corpus spirals as rv side labels.
# Per scroll:
#   1. our surface labels (scripts/v2/surflabels.sh via the labq.sh settings) into $LAB/S (full phase: 100-150 GB, temporary);
#   2. ufsm gt-gate on OUR labels vs the scan's m7 prob (mirror, cached): a 64^3 cell is bad when our sheet direction (phase gradient)
#      and the CT sheet direction (m7 structure tensor) differ by more than ORIENT (25) degrees: sheets veering off the CT. (The 11-06
#      support-radius gate failed both ways: our labels sit on recto faces 2-3 voxels from the m7 ridge, and tangles sit on real texture.)
#   3. the gated phase (bad cells + surf-labels ignore -> unknown), lossless, only shards keeping >= MINKEEP of their sheet voxels,
#      a SHFRAC hash subsample of them (disk), to $OUT/S/phase.zarr/<um>; gate mask $OUT/S/gate; then $LAB/S is deleted (KEEPFULL=1 keeps).
# Default list: every scroll with snapped wraps except PHercParis4 (own GT), PHerc1667 (eval only), PHerc0814 (skips sheets).
# Lanes: several good.sh may run at once; each claims a scroll with mkdir $OUT/S.claim (remove stale claims by hand).
# Env: U (ufsm with gt-gate --phase), TH (8), OUT (/mnt/scratch/rv/good), LAB (/mnt/scratch/rv/labels9), MINKEEP (0.3), SHFRAC (1),
#      Q (log /vesuvius/ufsm/v2/rv/logs/good.log). Skips scrolls with $OUT/S/done; stop file /vesuvius/ufsm/v2/rv/good.stop.
set -u
H=$(cd "$(dirname "$0")/.." && pwd); V2=/vesuvius/ufsm/v2; TSV=$V2/scrolls.tsv
U=${U:-/vesuvius/ufsm/v3/ps-src/build/ufsm}; OUT=${OUT:-/mnt/scratch/rv/good}; LAB=${LAB:-/mnt/scratch/rv/labels9}; Q=${Q:-$V2/rv/logs/good.log}
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; C=/mnt/scratch/vesuvius-offload/cache
L=${1:-$(ls $V2/wraps_snap | grep -v -E '^(STOP|PHercParis4|PHerc1667|PHerc0814)$')}
for S in $L; do
  [ -e $OUT/$S/done ] && continue
  mkdir -p $OUT; mkdir $OUT/$S.claim 2>/dev/null || continue   # (lanes: one claim per scroll)
  [ -f $V2/rv/good.stop ] && { echo "$(date) stop file" >> $Q; break; }
  row=$(awk -F'\t' -v s=$S '$1 == s' $TSV); [ -n "$row" ] || { echo "$(date) $S: not in $TSV" >> $Q; continue; }
  IFS=$'\t' read -r _ _ _ _ _ _ _ AX _ CTK UM LV _ <<< "$row"
  gum=$(awk -v u=$UM -v l=$LV 'BEGIN {printf "%.10g", u * 2 ^ l}'); AXS=$(awk -v l=$LV 'BEGIN {printf "%.10g", 2 ^ -l}')
  # the m7 prob of the same scan, at the label grid's um
  scan=$(basename "$CTK" | cut -d- -f1); PR=$M/$S/representations/predictions/surfaces
  pz=$(curl -s "$PR/" | grep -o "href=\"$scan-surface-[^\"]*-m7-L[0-9]-prob.zarr/\"" | head -1 | sed 's/href="//; s|/"||')
  [ -n "$pz" ] || { echo "$(date) $S: no m7 prob for scan $scan, skipped" >> $Q; continue; }
  pk=$($U info "$PR" "$pz" 2>/dev/null | awk -v g=$gum 'NR > 1 && $2 > 0 && ($2 / g > 0.99 && $2 / g < 1.01) {print $1; exit}')
  [ -n "$pk" ] || { echo "$(date) $S: m7 $pz has no level at $gum um, skipped" >> $Q; continue; }
  # 1. full labels (labq.sh settings), unless already there
  # a label build of S already running (an earlier lane): wait for it instead of starting a second one
  while [ ! -e $LAB/$S/done ] && pgrep -f "surflabels[.]sh $S " > /dev/null; do sleep 60; done
  if [ ! -e $LAB/$S/done ]; then
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); while [ "$avail" -lt 45 ]; do sleep 120; avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); done
    # disk: a full label set is 120-170 GB until gated and deleted; wait for MINDISK (250) GB free on $LAB
    while [ "$(df -BG --output=avail $LAB | tail -1 | tr -dc 0-9)" -lt ${MINDISK:-250} ]; do sleep 300; done
    echo "$(date) $S labels start" >> $Q
    OUT=$LAB U=$U TH=${TH:-8} bash $H/rv/labq.sh $S
    [ -e $LAB/$S/done ] || { echo "$(date) $S: surf-labels failed (see $LAB/$S/surflabels.log)" >> $Q; continue; }
  fi
  key=$(ls $LAB/$S/phase.zarr | grep -v zarr.json | head -1)
  [ "$key" = "$gum" ] || echo "$(date) $S: note label key $key vs grid $gum" >> $Q
  # 2-3. gate + gated phase
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); while [ "$avail" -lt 45 ]; do sleep 120; avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); done
  O=$OUT/$S; rm -rf $O; mkdir -p $O
  echo "$(date) $S gate start (m7 $pz/$pk, axis scale $AXS)" >> $Q
  nice $U gt-gate --gt $LAB/$S --gt-key codes.zarr/$key --prob $PR --prob-key $pz/$pk --cache $C --axis $AX --axis-scale $AXS \
    --out $O/gate --min-support 0 --orient-deg ${ORIENT:-25} --phase $LAB/$S --phase-key phase.zarr/$key --ignore $LAB/$S --ignore-key ignore.zarr/$key \
    --phase-out $O/phase.zarr --min-keep ${MINKEEP:-0.3} --shard-frac ${SHFRAC:-1} --jobs 16 > $O/gate.log 2>&1 || { echo "$(date) $S: gt-gate failed" >> $Q; continue; }
  cp $LAB/$S/surflabels.log $LAB/$S/phasecheck.txt $O/ 2>/dev/null
  echo "$S $key $(grep -o 'holding [0-9.]*' $O/gate.log) $(grep -o '[0-9]* shards written' $O/gate.log) $(du -sh $O/phase.zarr | cut -f1)" > $O/summary.txt
  touch $O/done; echo "$(date) $S done: $(cat $O/summary.txt)" >> $Q
  [ "${KEEPFULL:-0}" = 1 ] || rm -rf $LAB/$S
done

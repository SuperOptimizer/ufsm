#!/usr/bin/env bash
# scripts/v2/fetchct.sh [LANE NLANES]: v2 training-level CT (8.64/9.36 um L0, 2.4 um scans L2, plus levels L+2.. as scripts/u0/copy.sh)
# for every train + holdout scroll of scrolls.tsv (V2/scrolls.tsv, or scripts/corpus/list.tsv's scan when absent) -> V2/ct/<S>/<volume>.
# Scans already on the desk are symlinked (LOCAL: P3, P4, 0490B), never copied. Copy rows come from the u0 inventory sizes
# (INV, default /vesuvius/ufsm/u0/inventory-all.tsv) and go through scripts/u0/copy.sh (resumable ufsm mirror, byte-identical shards,
# done markers in V2/fetch/state, STOP file V2/fetch/STOP, live rate cap V2/fetch/mbps). Lanes split the copy rows round-robin;
# each lane's mirror keeps UFSM_HTTP_INFLIGHT (default 48 / NLANES) requests in flight (the mirror 429s past 64 per IP).
# INK=1: only the extra CT of the human ink sources (scripts/v2/inkscans.txt).
# Env: V2 (/vesuvius/ufsm/v2), U (V2/src/build/ufsm), MINFREE (GB kept free on V2's disk, 100), TH (mirror threads per lane, 8).
set -u
LANE=${1:-0}; NL=${2:-1}
H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $H/../.. && pwd)
V2=${V2:-/vesuvius/ufsm/v2}; U=${U:-$V2/src/build/ufsm}; INV=${INV:-/vesuvius/ufsm/u0/inventory-all.tsv}
A=$V2/fetch; mkdir -p $A/state $A/logs $V2/ct
# local scans: scroll -> dir holding <volume>/ (symlinked as V2/ct/<S>/<volume>)
declare -A LOCAL=([PHercParis3]=/vesuvius/ufsm/u0/vol/PHercParis3 [PHercParis4]=/vesuvius/ufsm/volcomp/PHercParis4 [PHerc0490B]=/vesuvius/ufsm/u0/vol/PHerc0490B)
EXCL=" PHerc0332 PHercMANB "
list=$A/list-$LANE${INK:+-ink}.tsv; : > $list; n=0
while IFS=$'\t' read -r S R K UM LV AX; do
  [ "$S" = scroll ] && continue; case "$EXCL" in *" $S "*) continue;; esac
  vol=$(basename $K)
  if [ -n "${LOCAL[$S]:-}" ]; then
    [ $LANE = 0 ] && { mkdir -p $V2/ct/$S; [ -e $V2/ct/$S/$vol ] || ln -s ${LOCAL[$S]}/$vol $V2/ct/$S/$vol; }
    continue; fi
  row=$(awk -F'\t' -v s=$S -v v=$vol '$1 == s && $3 == v' $INV | head -1)
  [ -n "$row" ] || { echo "fetchct: $S $vol not in $INV"; continue; }
  n=$((n + 1)); [ $(( (n - 1) % NL )) = $LANE ] || continue
  # copy.sh columns: scroll id volume um kev dist level lum shape lgb cgb shards status newgb pairs (lgb / cgb in GB)
  echo "$row" | awk -F'\t' 'BEGIN { OFS = "\t" } { print $1, $2, $3, $4, $5, $6, $7, $9, $10, $11 / 1e9, $13 / 1e9, $12, "copy", 0, "-" }' >> $list
done < <([ "${INK:-0}" = 1 ] || cat $REPO/scripts/corpus/list.tsv)
# INK=1: the extra scans of the human ink sources (inkscans.txt: scroll, volume), training level as the inventory says
if [ "${INK:-0}" = 1 ]; then while IFS=$'\t' read -r S vol; do
  row=$(awk -F'\t' -v s=$S -v v=$vol '$1 == s && $3 == v' $INV | head -1); [ -n "$row" ] || { echo "fetchct: $S $vol not in $INV"; continue; }
  n=$((n + 1)); [ $(( (n - 1) % NL )) = $LANE ] || continue
  echo "$row" | awk -F'\t' 'BEGIN { OFS = "\t" } { print $1, $2, $3, $4, $5, $6, $7, $9, $10, $11 / 1e9, $13 / 1e9, $12, "copy", 0, "-" }' >> $list
done < $H/inkscans.txt; fi
echo "$(date '+%F %T') fetchct lane $LANE/$NL: $(wc -l < $list) scans -> $V2/ct ($(df -BG --output=avail $V2 | tail -1 | tr -dc 0-9) GB free)"
A=$A U=$U VOL1=$V2/ct MINFREE1=${MINFREE:-100} TH=${TH:-8} UFSM_HTTP_INFLIGHT=${UFSM_HTTP_INFLIGHT:-$((48 / NL))} \
  bash $REPO/scripts/u0/copy.sh $list

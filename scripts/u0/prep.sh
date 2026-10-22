#!/usr/bin/env bash
# scripts/u0/prep.sh LIST: ufsm scanprep of every copied scan in LIST (inventory.tsv rows) -> A/prep/<scroll>-<id>.{json,mask.raw,
# mask.raw.json,mask.pgm}. Scans whose copy is not done are skipped (rerun later); restartable (done markers), nice 19, STOP file.
# Paris 4 (2.4 um, 20260411134726) and the 1667 test scan read the existing local trees under OLD (status use-*).
set -u
LIST=${1:?inventory rows}
A=${A:-/vesuvius/ufsm/u0}; U=${U:-$A/bin/ufsm}; OLD=${OLD:-/vesuvius/ufsm/volcomp}; TH=${TH:-8}
mkdir -p $A/prep $A/state $A/logs
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
while IFS=$'\t' read -r scroll id volume um kev dist level lum shape lgb cgb shards status newgb pairs; do
  [ "$scroll" = scroll ] && continue
  [ -e $A/STOP ] && { log "STOP file: exiting"; exit 0; }
  tag=$scroll-$id
  case "$status" in
    copy|link-old-copy) [ -e $A/state/$tag.copy.done ] || continue; root=$(cat $A/state/$tag.loc 2>/dev/null || echo $A/vol); key=$scroll/$volume;;
    use-*) root=$OLD/$scroll; key=$volume;;
    *) continue;;
  esac
  [ -e $A/state/$tag.prep.done ] && continue
  MI=${MINSIDE:-0.25}
  extra=$(awk -v t=$tag '$1 == t {$1 = ""; print}' $A/scripts/prep-overrides.tsv 2>/dev/null)   # per-scan scanprep options (e.g. --zrange)
  if nice -n 19 $U scanprep $root $key --level $level --um $um --out $A/prep/$tag --min-inside $MI --threads $TH $extra > $A/logs/$tag-prep.log 2>&1; then
    date '+%F %T' > $A/state/$tag.prep.done; log "$tag: $(grep -h 'object cells' $A/logs/$tag-prep.log | sed 's/^scanprep: //')"
  else log "$tag: scanprep FAILED (logs/$tag-prep.log)"; fi
done < "$LIST"
log "prep done"

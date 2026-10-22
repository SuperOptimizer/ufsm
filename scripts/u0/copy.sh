#!/usr/bin/env bash
# scripts/u0/copy.sh LIST: local copies (ufsm mirror) of the ~9.6 um level of each scan in LIST (inventory.tsv rows: status copy or
# link-old-copy) on the desk NVMe, under A/vol/<scroll>/<volume>: level L (the training level), levels L+2.. (scanprep's cell means,
# the sampler's coarse probe at L+3), a group zarr.json listing only those levels. Byte-identical 1024^3 shards of the mirror.
# Restartable (done markers, mirror skips present shards), nice 19 + idle IO class, --mbps cap, disk guard (stops below MINFREE GB
# after the scan), STOP file. Run: cd A; setsid nohup bash scripts/copy.sh LIST > logs/copy-N.log 2>&1 < /dev/null &
set -u
LIST=${1:?inventory rows}
A=${A:-/vesuvius/ufsm/u0}; U=${U:-$A/bin/ufsm}; R=${R:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
OLD=${OLD:-/vesuvius/ufsm/volcomp}; MBPS=${MBPS:-150}; TH=${TH:-6}; MINFREE=${MINFREE:-250}
mkdir -p $A/state $A/logs $A/vol
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
while IFS=$'\t' read -r scroll id volume um kev dist level lum shape lgb cgb shards status newgb pairs; do
  [ "$scroll" = scroll ] && continue
  case "$status" in copy|link-old-copy) ;; *) continue;; esac
  [ -e $A/STOP ] && { log "STOP file: exiting"; exit 0; }
  tag=$scroll-$id; D=$A/vol/$scroll/$volume
  [ -e $A/state/$tag.copy.done ] && continue
  need=$(awk -v a="$lgb" -v b="$cgb" 'BEGIN{printf "%d", a + b + 1}')
  free=$(df -BG --output=avail $A | tail -n 1 | tr -dc 0-9)
  if [ $((free - need)) -lt $MINFREE ]; then log "$tag: needs ${need} GB, ${free} GB free: below the ${MINFREE} GB floor, stopping"; exit 3; fi
  nlev=$(curl -sf --retry 3 "$R/$scroll/volumes/$volume/zarr.json" | jq '.attributes.ome.multiscales[0].datasets | length')
  [ -n "$nlev" ] || { log "$tag: no group metadata"; continue; }
  levels="$level"; for ((l = level + 2; l < nlev; l++)); do levels="$levels $l"; done
  log "$tag: levels $levels (${need} GB, ${free} GB free)"
  ok=1
  for l in $levels; do
    lk=(); [ -d "$OLD/$scroll/volumes/$volume/$l/c" ] && lk=(--link-from "$OLD/$scroll/volumes/$volume/$l")
    nv=2; [ $l = $level ] && nv=8
    nice -n 19 ionice -c 3 $U mirror $R $scroll/volumes/$volume/$l $D/$l --threads $TH --mbps $MBPS "${lk[@]}" --verify $nv --seed $l \
      > $A/logs/$tag-L$l.log 2>&1 || { ok=0; log "$tag: level $l FAILED (logs/$tag-L$l.log)"; }
    tail -n 2 $A/logs/$tag-L$l.log | sed "s/^/  /"
  done
  # the group metadata with only the copied levels (the sampler then never retries an absent level), scales in absolute um (the
  # mirror's are relative 1, 2, 4 ..: ufsm matches relative scales only when the first listed level is 1)
  curl -sf --retry 3 "$R/$scroll/volumes/$volume/zarr.json" |
    jq --argjson keep "[$(echo $levels | sed 's/ /,/g; s/[0-9]\+/"&"/g')]" \
       --argjson um "$um" '.attributes.ome.multiscales[0].datasets |= map(select(.path as $p | $keep | index($p)) | .coordinateTransformations[0].scale |= map(. * $um))' > $D/zarr.json.part && mv $D/zarr.json.part $D/zarr.json || ok=0
  [ $ok = 1 ] && { date '+%F %T' > $A/state/$tag.copy.done; log "$tag: done ($(du -sh $D | cut -f1))"; }
done < "$LIST"
log "list done"

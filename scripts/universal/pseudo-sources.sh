#!/usr/bin/env bash
# scripts/universal/pseudo-sources.sh PSDIR U0.json GTW > pseudo.json: the wind96 pseudo-label boxes (scripts/dino/pseudo-scan.sh outputs
# copied to PSDIR: PSDIR/<scan>/sources.json + b<i>/codes) as labelled sources of the universal run (jq only)
#   - each box's scan resolved against the U0 sources (same scan: root / ct group / level / scanprep norm), codes.root re-rooted under PSDIR
#   - boxes with < MINPCT % labelled dropped; boxes touching a holdout of their U0 scan dropped (the GT eval boxes stay clean)
#   - weights by labelled fraction; the pseudo group's total = PW x GTW (GTW: the real GT sources' total weight)
#   CS: sources (default gate 0.5,0.1,0.3) or sources2 (the loose 0.3,0.15,0.3 set)
set -eu
PS=${1:?pseudo dir}; U0=${2:?u0 sources json}; GTW=${3:?real GT total weight}; MINPCT=${MINPCT:-2}; PW=${PW:-1}; CS=${CS:-sources}
PS=$(cd $PS && pwd)
cat $PS/*/$CS.json | jq -s --arg ps "$PS" --argjson minp $MINPCT --argjson pw $PW --argjson gtw $GTW --slurpfile u0 $U0 '
  def isect($a; $b): all(range(3); $a[.] < $b[.] + $b[. + 3] and $b[.] < $a[.] + $a[. + 3]);
  # the U0 entry of a pseudo box: same scroll and scan id ("S/volumes/<id>-....zarr/L" vs "S/<id>-....zarr" + min_level)
  def scankey: sub("/volumes/"; "/") | sub("\\.zarr(/[0-9]+)?$"; "");
  ($u0[0].sources | map({key: (.ct | sub("\\.zarr$"; "")), value: .}) | from_entries) as $U
  | [.[][] | (.ct | scankey) as $k | ($U[$k] // ($U[$k | sub("^[^/]+/"; "")])) as $e | select($e != null)
       | select(.pseudo.labelled_pct >= $minp)
       | select(([.boxes[] as $b | ($e.holdout // [])[] | isect($b; .)] | any) | not)
       | {name, root: $e.root, ct: $e.ct, min_level: $e.min_level, max_level: $e.max_level, um: $e.um, norm: $e.norm, boxes, origin,
          codes: (.codes + {root: ($ps + (.codes.root | sub(".*/pseudo/scans"; "")))}), pseudo}] as $l
  | ($l | map(.pseudo.labelled_pct) | add // 1) as $s
  | {sources: [$l[] | . + {weight: ($pw * $gtw * .pseudo.labelled_pct / $s)}]}'

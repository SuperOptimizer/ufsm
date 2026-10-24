#!/usr/bin/env bash
# scripts/universal/pseudo-sources.sh PSDIR U0.json GTW > pseudo.json: the wind96 pseudo-label boxes (scripts/dino/pseudo-scan.sh outputs
# copied to PSDIR: PSDIR/<scan>/sources.json + b<i>/codes) as labelled sources of the universal run (jq only)
#   - each box's scan resolved against the U0 sources (same scan: root / ct group / level / scanprep norm), codes.root re-rooted under PSDIR
#   - boxes with < MINPCT % labelled dropped; boxes touching a holdout of their U0 scan dropped (the GT eval boxes stay clean)
#   - weights by labelled fraction; the pseudo group's total = PW x GTW (GTW: the real GT sources' total weight)
#   SRAD (default 96): "supcon_radius" of every pseudo box (CT voxels): SupCon pairs only within that distance (the pseudo codes are gauge-
#     consistent per ~128^3 (0.83-0.94) but not box-wide (0.64): a P512 window would feed contradictory long-range pairs); 0 = all pairs
#   CS: sources2 (default: the loose 0.3,0.15,0.3 gate; on 0139 its per-128^3 accuracy vs the GT equals the strict gate's, 0.836 vs 0.832) or sources (0.5,0.1,0.3)
set -eu
PS=${1:?pseudo dir}; U0=${2:?u0 sources json}; GTW=${3:?real GT total weight}; MINPCT=${MINPCT:-2}; PW=${PW:-1}; CS=${CS:-sources2}; SRAD=${SRAD:-96}
PS=$(cd $PS && pwd)
cat $PS/*/$CS.json | jq -s --arg ps "$PS" --argjson minp $MINPCT --argjson pw $PW --argjson gtw $GTW --argjson srad $SRAD --slurpfile u0 $U0 '
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
  | {sources: [$l[] | . + {weight: ($pw * $gtw * .pseudo.labelled_pct / $s), supcon_radius: $srad}]}'

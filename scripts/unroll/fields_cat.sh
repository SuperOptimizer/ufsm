#!/usr/bin/env bash
# fields_cat.sh OUT A B: one fit-field directory (zarr format, predict --fit-zarr) from two z parts A (below) and B (above) predicted
# separately, e.g. one per GPU: predict --box with the same y / x and B's z origin = A's origin + A's cells x h, on the same global
# tile grid (--grid-origin), so every window, and therefore every field shard, is the one a single run computes. A's z cells must be a
# multiple of the field shard (128 cells). Shard files are MOVED (same filesystem: instant), A and B are emptied and removed; OUT's
# zarr.json shapes cover both, fields.json joins the axis rows. No data is rewritten.
set -eu
out=$1 a=$2 b=$3
[ -e "$out" ] && { echo "fields_cat: $out exists"; exit 1; }
for d in "$a" "$b"; do [ "$(jq -r .format "$d/fields.json")" = zarr ] || { echo "fields_cat: $d is not a zarr fit-field dir"; exit 1; }; done
read -r az ay ax ah aZ aY aX < <(jq -r '[.origin[], .h, .shape[]] | @tsv' "$a/fields.json")
read -r bz by bx bh bZ bY bX < <(jq -r '[.origin[], .h, .shape[]] | @tsv' "$b/fields.json")
[ "$ah" = "$bh" ] && [ "$ay" = "$by" ] && [ "$ax" = "$bx" ] && [ "$aY" = "$bY" ] && [ "$aX" = "$bX" ] || { echo "fields_cat: y / x boxes or h differ"; exit 1; }
[ $bz -eq $((az + aZ * ah)) ] || { echo "fields_cat: B starts at z $bz, A ends at $((az + aZ * ah))"; exit 1; }
cs=$(jq -r '.chunk_grid.configuration.chunk_shape[0]' "$a/pc/zarr.json")
[ $((aZ % cs)) -eq 0 ] || { echo "fields_cat: A has $aZ z cells, not a multiple of the shard $cs"; exit 1; }
off=$((aZ / cs)); n=$((aZ + bZ))
mkdir -p "$out"
for p in $(ls "$a"); do
  [ -d "$a/$p" ] && [ -e "$a/$p/zarr.json" ] || continue
  [ -e "$b/$p/zarr.json" ] || { echo "fields_cat: $b has no $p"; exit 1; }
  mkdir -p "$out/$p/c"
  jq -c --argjson n $n '.shape[0] = $n' "$a/$p/zarr.json" > "$out/$p/zarr.json"
  [ -d "$a/$p/c" ] && for z in $(ls "$a/$p/c"); do mv "$a/$p/c/$z" "$out/$p/c/$z"; done
  [ -d "$b/$p/c" ] && for z in $(ls "$b/$p/c" | sort -rn); do mv "$b/$p/c/$z" "$out/$p/c/$((z + off))"; done
done
# axis: A's rows z 0 .. aZ h - 1, then B's (its first row is the same z as A's last + 1 .. its end)
jq -c --argjson n $n --argjson k $((aZ * ah)) --slurpfile b "$b/fields.json" '.shape[0] = $n | .axis = (.axis[0:$k] + $b[0].axis)' "$a/fields.json" > "$out/fields.json"
for d in "$a" "$b"; do find "$d" -depth -type d -empty -delete; rm -f "$d/fields.json"; for p in $(ls "$d" 2>/dev/null); do rm -f "$d/$p/zarr.json"; done; find "$d" -depth -type d -empty -delete; done
echo "fields_cat: $out: $(jq -c '{origin, shape, h, na: (.axis | length)}' "$out/fields.json")"

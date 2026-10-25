#!/usr/bin/env bash
# mkcrop.sh SRC OUT Z Y X NZ NY NX: a fit-field directory OUT for the box of field cells [Z, Z+NZ) x [Y, Y+NY) x [X, X+NX) of the
# fit-field directory SRC (zarr or raw arrays), without copying: OUT's arrays are symlinks, its fields.json carries the box (origin,
# shape, the axis rows of the box made box-local) and the array offset ("zarr_offset" / "raw_offset" + "raw_shape", src/fit.c).
set -eu
src=$(readlink -f "$1") out=$2; z=$3 y=$4 x=$5 nz=$6 ny=$7 nx=$8
mkdir -p "$out"
fmt=$(jq -r '.format // "raw"' "$src/fields.json")
for a in pc ps t0 t1 t2 t3 t4 t5 sp w dz dy dx dq r; do   # (the winding mod 4 planes and the recto plane when the source has them)
  if [ "$fmt" = zarr ]; then [ -e "$src/$a" ] && ln -sfn "$src/$a" "$out/$a"; else [ -e "$src/$a.raw" ] && ln -sfn "$src/$a.raw" "$out/$a.raw"; fi
done
jq -c --argjson z "$z" --argjson y "$y" --argjson x "$x" --argjson nz "$nz" --argjson ny "$ny" --argjson nx "$nx" --arg fmt "$fmt" '
  .h as $h | (.zarr_offset // [0,0,0]) as $zo | (.raw_offset // [0,0,0]) as $ro | (.raw_shape // .shape) as $rs |
  {format: $fmt, origin: [.origin[0] + $z * $h, .origin[1] + $y * $h, .origin[2] + $x * $h], shape: [$nz, $ny, $nx], h: $h, um: .um,
   axis: [.axis[$z * $h : ($z + $nz) * $h + 1][] | [.[0] - $y * $h, .[1] - $x * $h]]}
  + (if .w4_d then {w4_d: .w4_d} else {} end) + (if .recto != null then {recto: .recto} else {} end)
  + (if $fmt == "zarr" then {zarr_offset: [$zo[0] + $z, $zo[1] + $y, $zo[2] + $x]}
     else {raw_offset: [$ro[0] + $z, $ro[1] + $y, $ro[2] + $x], raw_shape: $rs} end)' "$src/fields.json" > "$out/fields.json"
echo "$out: $(jq -c '{origin, shape, h, um, zarr_offset, raw_offset, axis0: .axis[0], na: (.axis | length)}' "$out/fields.json")"

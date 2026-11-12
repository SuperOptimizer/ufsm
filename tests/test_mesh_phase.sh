#!/usr/bin/env bash
# sources "phase": {"onfly": {"meshes": {...}, "rmax"}}: the pair-rule phase from winding codes rasterised on read from q-assigned meshes
# (meshcodes.h), behind the derived-array cache. Four meshes (tests/test_codes_onfly.sh's: two nested sheets a turn apart, a crossing
# plane, holes / NaN q); `ufsm raster` writes the stored level-0 and level-1 codes and a 2-level CT pyramid (the grid meta).
# test_mesh_phase: the meshes' phase == the stored codes' phase at level 0 and at level 1 (the next rung: codes rasterised at level 1),
# and a second load is served from the disk cache.
# Usage: test_mesh_phase.sh UFSM TEST_MESH_PHASE DIR
set -eu
U=$(readlink -f "$1"); T=$(readlink -f "$2"); D=$3; SRC=$(dirname "$(readlink -f "$0")")
rm -rf "$D"; mkdir -p "$D/m"; cd "$D"
sed -n '/^python3 - <<.PY.$/,/^PY$/p' "$SRC/test_codes_onfly.sh" > gen.sh; bash gen.sh
M=$(ls -d m/*.tifxyz)
for L in 0 1; do
  "$U" raster st$L --shape 256,320,320 --um 2.4 --level $L --binary 0 --levels 1 --threads 4 --shard 128 --value q.tif $M > r$L.log 2>&1 || { echo "FAIL: raster level $L ($D/r$L.log)"; exit 1; }
done
"$U" raster ct --shape 256,320,320 --um 2.4 --levels 2 --threads 4 --shard 128 $M > rc.log 2>&1 || { echo "FAIL: raster ct ($D/rc.log)"; exit 1; }
P=$(pwd)
cat > mesh.json <<J
{"sources": [{"name": "mesh", "root": "$P", "ct": "ct", "um": 2.4, "targets": {"band": {"phase": {"level": 0,
  "onfly": {"meshes": {"dir": "$P/m", "value": "q.tif"}, "rmax": 12, "cache_dir": "$P/dc"}}}}}]}
J
cat > st0.json <<J
{"sources": [{"name": "st0", "root": "$P", "ct": "ct", "um": 2.4, "targets": {"band": {"phase": {"level": 0, "onfly": {"root": "$P/st0", "key": "2.4", "rmax": 12}}}}}]}
J
cat > st1.json <<J
{"sources": [{"name": "st1", "root": "$P", "ct": "ct", "um": 2.4, "targets": {"band": {"phase": {"level": 1, "onfly": {"root": "$P/st1", "key": "4.8", "rmax": 6}}}}}]}
J
UFSM_MESHCODES_CACHE="$P/mc" "$T" mesh.json st0.json st1.json > t.log 2>&1 || { echo "FAIL: test_mesh_phase ($D/t.log)"; tail -6 t.log; exit 1; }
grep -E 'differ|dcache' t.log
echo "test_mesh_phase: OK"

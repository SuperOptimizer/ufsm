#!/usr/bin/env bash
# train --strict-sources 1 and the pyramid reader's relative-scale groups (2026-11-12: a mirror json with relative scales over a group
# without level 0 made the reader take 2.0 as um, and PHerc1667 was silently excluded from u1 for 315 steps):
#  (1) a weighted source that cannot draw (min_level 3: 32^3 < P 64) logs "excluded" and trains without --strict-sources; with
#      --strict-sources 1 the run stops at startup with "fatal";
#  (2) the CT group's json rewritten to relative scales (2, 4, 8, 16: level 0 not listed) is resolved as relative: a level-1 source on it
#      trains under --strict-sources 1, and its level-1 window reads the same CT as the absolute-um group.
# Usage: test_strict_sources.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
C=(--task unwrap2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --steps 2)
python3 -I - <<'PY'
import json, copy, os, shutil
S = json.load(open('fx/sources.json')); s0 = S['sources'][0]
bad = copy.deepcopy(s0); bad['name'] = 'too-coarse'; bad['min_level'] = bad['max_level'] = 3
json.dump({'sources': [s0, bad]}, open('two.json', 'w'))
# the relative-scale group over the same arrays (symlinked level dirs, no level 0)
os.makedirs('rel', exist_ok=True); m = json.load(open('fx/ct/zarr.json')); ms = m['attributes']['ome']['multiscales'][0]
ds = []
for d in ms['datasets'][1:]:
    f = round(d['coordinateTransformations'][0]['scale'][0] / 9.6); os.symlink(os.path.abspath('fx/ct/' + d['path']), 'rel/' + d['path'])
    d = copy.deepcopy(d); d['coordinateTransformations'][0]['scale'] = [f, f, f]; ds.append(d)
ms['datasets'] = ds; json.dump(m, open('rel/zarr.json', 'w'))
r = copy.deepcopy(s0); r['name'] = 'relative'; r['root'] = '.'; r['ct'] = 'rel'; r['min_level'] = r['max_level'] = 1
r['targets']['band']['phase'] = {'root': 'fx/phase', 'key': '1', 'level': 1}
json.dump({'sources': [r]}, open('rel.json', 'w'))
r = copy.deepcopy(r); r['name'] = 'absolute'; r['root'] = 'fx'; r['ct'] = 'ct'; r['targets']['band']['phase']['root'] = 'fx/phase'
json.dump({'sources': [r]}, open('abs.json', 'w'))
PY
"$U" train two.json --out a --levels 1 "${C[@]}" > a.log 2>&1 || fail "without --strict-sources ($D/a.log)"
grep -q 'too-coarse excluded' a.log || fail "no exclusion logged ($D/a.log)"
if "$U" train two.json --out b --levels 1 --strict-sources 1 "${C[@]}" > b.log 2>&1; then fail "--strict-sources 1 did not stop ($D/b.log)"; fi
grep -q 'too-coarse excluded.*fatal' b.log || fail "no fatal line ($D/b.log)"
[ -d fx/phase/1 ] || fail "the fixture has no fx/phase/1"
"$U" train rel.json --out c --strict-sources 1 --levels 0,1 "${C[@]}" > c.log 2>&1 || fail "relative-scale group ($D/c.log)"
grep -q '^step      2' c.log || fail "relative-scale group: no step 2 ($D/c.log)"
"$U" train abs.json --out d --strict-sources 1 --levels 0,1 "${C[@]}" > d.log 2>&1 || fail "absolute-um group ($D/d.log)"
lc=$(grep '^step      1' c.log | grep -o 'loss [0-9.]*'); ld=$(grep '^step      1' d.log | grep -o 'loss [0-9.]*')
[ -n "$lc" ] && [ "$lc" = "$ld" ] || fail "relative vs absolute group: step-1 $lc vs $ld"
echo "relative-scale group (2 4 8 16, no level 0): level-1 step-1 $lc == the absolute-um group's"
echo "test_strict_sources: ok"

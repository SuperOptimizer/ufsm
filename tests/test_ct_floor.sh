#!/usr/bin/env bash
# The per-scan air iso (DESIGN 2026-11-17 "per-scroll air iso"):
#  (1) ufsm ct-iso --from-hist on a synthetic two-mode histogram (air ~45, papyrus ~115, a constant-fill spike at 28 and masked zeros):
#      air / pap / valley / mid where they were put; one without an air mode: method no-air, exit 3; ct-iso on the fixture CT runs
#  (2) predict --ct-floor T: the CT windows the net sees (UFSM_PRED_CTDUMP) are the unfloored windows with every voxel below T set to 0,
#      the output changes, the host path (UFSM_PRED_HOSTPATH=1, floored in the reader; a voxel net: latent-grid nets have no host path) writes the same bytes as the
#      device path, the store
#      records ct_floor; --ct-floor 300 is refused
# Usage: test_ct_floor.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
# ---- (1) ----
python3 -I - <<'PY'
import math
def g(m, s, a): return [a * math.exp(-0.5 * ((i - m) / s) ** 2) for i in range(256)]
air, pap = g(45, 5, 1e6), g(115, 18, 6e5)
h = [round(air[i] + pap[i]) for i in range(256)]; h[0] = 10 ** 8; h[28] += 4 * 10 ** 6
open('two.txt', 'w').write(' '.join(map(str, h[:128] + [0] * 128)) + '\n' + ' '.join(map(str, [0] * 128 + h[128:])) + '\n')   # (two lines: summed)
open('one.txt', 'w').write(' '.join(str(round(pap[i])) for i in range(256)) + '\n')
PY
"$U" ct-iso --from-hist two.txt --scroll syn > two.tsv || fail "ct-iso two-mode"
IFS=$'\t' read -r s k l air pap val mid dep nb meth < two.tsv
echo "  two-mode: air $air pap $pap valley $val mid $mid depth $dep boxes $nb $meth"
[ "$air" = 45 ] && [ "$pap" = 115 ] && [ "$val" -gt 60 ] && [ "$val" -lt 80 ] && [ "$mid" = 80 ] && [ "$nb" = 2 ] && [ "$meth" = valley ] || fail "two-mode: $(cat two.tsv)"
set +e; "$U" ct-iso --from-hist one.txt > one.tsv; rc=$?; set -e
[ "$rc" = 3 ] && grep -q 'no-air' one.tsv || fail "one mode: exit $rc, $(cat one.tsv)"
"$MF" fx 256 2 1 > fx.log 2>&1
set +e; "$U" ct-iso fx ct 9.6 --grid 2 --scroll fx --header 1 > fx.tsv 2> fxiso.log; rc=$?; set -e
[ "$rc" = 0 ] || [ "$rc" = 3 ] || fail "ct-iso on the fixture exited $rc ($D/fxiso.log)"; [ "$(wc -l < fx.tsv)" = 2 ] || fail "ct-iso fixture rows"
# ---- (2) ----
T=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 2 --lat 4)
"$U" train fx/sources.json --out a "${T[@]}" --seed 5 > a.log 2>&1 || fail "train ($D/a.log)"
P=(--um 9.6 --level 0 --gpu 0 --threads 2 --shard 128 --levels 1 --box 0,0,0,256,256,256 --window 160 --halo 16 --q 0 --no-axis 1)
UFSM_PRED_CTDUMP=c0.ct "$U" predict a/last.ckpt fx ct o0 "${P[@]}" > o0.log 2>&1 || fail "predict ($D/o0.log)"
F=$(python3 -I -c "import numpy as np; a = np.fromfile('c0.ct', np.uint8); print(int(np.median(a[a > 0])))")
echo "  floor $F (the fixture's median nonzero CT)"
UFSM_PRED_CTDUMP=c1.ct "$U" predict a/last.ckpt fx ct o1 "${P[@]}" --ct-floor $F > o1.log 2>&1 || fail "predict --ct-floor ($D/o1.log)"
# the host path (latent-grid nets run the device path only): a voxel net, device against host
"$U" train fx/sources.json --out v "${T[@]:0:${#T[@]}-2}" --seed 5 > v.log 2>&1 || fail "train voxel net ($D/v.log)"
"$U" predict v/last.ckpt fx ct o4 "${P[@]}" --ct-floor $F > o4.log 2>&1 || fail "predict --ct-floor, voxel net ($D/o4.log)"
UFSM_PRED_HOSTPATH=1 "$U" predict v/last.ckpt fx ct o2 "${P[@]}" --ct-floor $F > o2.log 2>&1 || fail "predict --ct-floor, host path ($D/o2.log)"
python3 -I - c0.ct c1.ct $F <<'PY' || fail "the floored CT windows"
import sys, numpy as np
a = np.fromfile(sys.argv[1], np.uint8); b = np.fromfile(sys.argv[2], np.uint8); t = int(sys.argv[3])
e = np.where(a < t, 0, a).astype(np.uint8)
print(f"  CT windows: {a.size} voxels, {np.count_nonzero(a)} nonzero, {np.count_nonzero(b)} after the floor, {np.count_nonzero(e != b)} differ from the expected")
sys.exit(0 if a.size == b.size and a.size > 0 and np.array_equal(e, b) and 0 < np.count_nonzero(b) < np.count_nonzero(a) else 1)
PY
"$U" read o0 9.6 0 0 0 256 256 256 a.raw > /dev/null 2>&1 && "$U" read o1 9.6 0 0 0 256 256 256 b.raw > /dev/null 2>&1 && "$U" read o2 9.6 0 0 0 256 256 256 c.raw > /dev/null 2>&1 && "$U" read o4 9.6 0 0 0 256 256 256 d.raw > /dev/null 2>&1 || fail "read outputs"
cmp -s a.raw b.raw && fail "--ct-floor $F changed no output byte"
cmp -s d.raw c.raw || fail "--ct-floor: the host path differs from the device path"
grep -q "\"ct_floor\":$F" o1/9.6/zarr.json || fail "no ct_floor in o1/9.6/zarr.json"
set +e; "$U" predict a/last.ckpt fx ct o3 "${P[@]}" --ct-floor 300 > o3.log 2>&1; rc=$?; set -e
[ "$rc" = 2 ] || fail "--ct-floor 300 exited $rc, not 2"
echo "PASS test_ct_floor"

#!/usr/bin/env bash
# train --scan-norm 1 --gn-ema 1 on the wind fixture: the sampler's per-scan CT normalisation (scannorm.h: mean / sd of the nonzero
# voxels in boxes picked on a coarse level) is reported, deterministic across runs, kept in the checkpoint with the running GroupNorm
# statistics, and predict on the same scan and level computes the same center / scale for the net (and --norm overrides it); the
# training input changes with it (same seed, --det 1: different losses). A latvol CT: the statistics of the rebuilt CT, or a refusal
# Usage: test_scan_norm.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
export UFSM_SCANNORM_DIR=$PWD/snc   # (the statistics' disk cache: run a computes, run b reads it back and verifies the bits)
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 4)
UFSM_BATCH_HASH=1 "$U" train fx/sources.json --out a "${C[@]}" --scan-norm 1 --gn-ema 1 --gn-ema-warm 2 > a.log 2>&1 || { echo "FAIL: train ($D/a.log)"; exit 1; }
UFSM_BATCH_HASH=1 UFSM_SCANNORM_VERIFY=1 "$U" train fx/sources.json --out b "${C[@]}" --scan-norm 1 --gn-ema 1 --gn-ema-warm 2 > b.log 2>&1 || { echo "FAIL: train again ($D/b.log)"; exit 1; }
"$U" train fx/sources.json --out c "${C[@]}" > c.log 2>&1 || { echo "FAIL: train without ($D/c.log)"; exit 1; }
la=$(grep -m1 -o 'scan-norm: level 0 center [0-9.]* scale [0-9.]*' a.log || true)
lb=$(grep -m1 -o 'scan-norm: level 0 center [0-9.]* scale [0-9.]*' b.log || true)
[ -n "$la" ] && [ "$la" = "$lb" ] || { echo "FAIL: sampler statistics missing or not deterministic ('$la' vs '$lb')"; exit 1; }
grep -q 'scan-norm: level 0 .*(computed' a.log && grep -q 'scan-norm: level 0 .*(cached' b.log && grep -q 'scannorm: verified' b.log \
  || { echo "FAIL: the statistics not computed in a / not read back from the disk cache and verified in b"; exit 1; }
[ "$(grep '^batchhash' a.log | sed -E 's/ slot [0-9]+//')" = "$(grep '^batchhash' b.log | sed -E 's/ slot [0-9]+//')" ] && grep -q '^batchhash' a.log \
  || { echo "FAIL: the batches differ with the cached statistics"; exit 1; }
grep -q 'GroupNorm: running statistics' a.log || { echo "FAIL: no gn_ema startup line"; exit 1; }
head -1 a/last.ckpt | grep -q '"scan_norm":1' || { echo "FAIL: checkpoint without scan_norm"; exit 1; }
head -1 a/last.ckpt | grep -Eq '"gn_ema":1,"ge_n":4,' || { echo "FAIL: checkpoint gn_ema / 4 updates"; head -1 a/last.ckpt | grep -o '"gn_ema":[^[]*' ; exit 1; }
for f in a c; do grep '^step' $f.log | sed -E 's/ [0-9.]+ samp\/s.*//' > $f.txt; done
cmp -s a.txt c.txt && { echo "FAIL: the losses did not change with --scan-norm / --gn-ema"; exit 1; }
"$U" predict a/last.ckpt fx ct pa --um 9.6 --level 0 --box 64,64,64,128,128,128 --gpu 0 --threads 2 --window 128 --halo 16 > pa.log 2>&1 || { echo "FAIL: predict ($D/pa.log)"; exit 1; }
lp=$(grep -m1 -o 'scan-norm net: level 0 center [0-9.]* scale [0-9.]*' pa.log || true)
[ "${lp#scan-norm net: }" = "${la#scan-norm: }" ] || { echo "FAIL: predict computes '$lp', the sampler '$la'"; exit 1; }
"$U" predict a/last.ckpt fx ct pn --um 9.6 --level 0 --box 64,64,64,128,128,128 --gpu 0 --threads 2 --window 128 --halo 16 --norm 100,40 > pn.log 2>&1 || { echo "FAIL: predict --norm ($D/pn.log)"; exit 1; }
grep -q 'scan-norm net' pn.log && { echo "FAIL: --norm did not override"; exit 1; }
# a latvol CT (unroll audit 11-14 #15): predict computes the scan statistics on the rebuilt CT (as the sampler does for a latvol source),
# close to the plain CT's; a latvol group without the coarse levels refuses instead of the silent per-window z-score. The transform: a
# Walsh-Hadamard basis / 8 (orthonormal, row 0 the cell mean), written here; C 64 (all coefficients) keeps the rebuilt CT near-lossless
python3 -I - lv.lcb <<'PY'
import struct, sys
H = [[1]]
while len(H) < 64: H = [r + r for r in H] + [r + [-v for v in r] for r in H]
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<2I', 0x424c4655, 1)); f.write(struct.pack('<64f', *[64.0 / (k + 1) for k in range(64)]))
    for r in H: f.write(struct.pack('<64f', *[v / 8 for v in r]))
PY
mkdir -p lv lv1
"$U" latvol lv.lcb fx ct lv/ct --levels 9.6,19.2,38.4,76.8,153.6 --c 64 --threads 2 --gpu -1 > lv.log 2>&1 || { echo "FAIL: latvol ($D/lv.log)"; exit 1; }
"$U" latvol lv.lcb fx ct lv1/ct --levels 9.6 --c 64 --threads 2 --gpu -1 > lv1.log 2>&1 || { echo "FAIL: latvol one level ($D/lv1.log)"; exit 1; }
UFSM_SCANNORM_CACHE=0 "$U" predict a/last.ckpt lv ct pl --um 9.6 --level 0 --box 64,64,64,128,128,128 --gpu 0 --threads 2 --window 128 --halo 16 > pl.log 2>&1 || { echo "FAIL: predict on latvol ($D/pl.log)"; exit 1; }
ll=$(grep -m1 -o 'scan-norm net: level 0 center [0-9.]* scale [0-9.]*' pl.log || true)
[ -n "$ll" ] && grep -q 'latvol CT: the rebuilt CT' pl.log || { echo "FAIL: no scan statistics on the latvol CT ($D/pl.log)"; exit 1; }
read -r c1 s1 c2 s2 <<< "$(echo "$lp $ll" | grep -o 'center [0-9.]* scale [0-9.]*' | awk '{printf "%s %s ", $2, $4}')"
awk -v a="$c1" -v b="$c2" -v x="$s1" -v y="$s2" 'BEGIN { d = a - b; e = x / y - 1; exit !(d < 2 && d > -2 && e < .05 && e > -.05) }' \
  || { echo "FAIL: latvol statistics '$ll' far from the CT's '$lp'"; exit 1; }
if UFSM_SCANNORM_CACHE=0 "$U" predict a/last.ckpt lv1 ct p1 --um 9.6 --level 0 --box 64,64,64,128,128,128 --gpu 0 --threads 2 --window 128 --halo 16 > p1.log 2>&1; then
  echo "FAIL: predict on a latvol group without coarse levels ran (per-window statistics, $D/p1.log)"; exit 1; fi
grep -q 'no statistics at level 0 of ct (latvol)' p1.log || { echo "FAIL: no refusal message ($D/p1.log)"; exit 1; }
echo "test_scan_norm: ok ($la; latvol $ll)"

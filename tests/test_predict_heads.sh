#!/usr/bin/env bash
# predict --heads (every head of a latent-grid net from one trunk pass, nn_lat_heads_place) on the wind fixture:
#  (1) a sub-cell net with a fine stem, rv 2 and ink (--lat 4 --lat-guide 3 --fstem 16): --heads all writes <out>/rv and <out>/ink; rv equals
#      the old slab path (UFSM_PRED_HEADS=0: upsampling + nn_fstem_fwd + nn_rv_fuse) within one code on 0.1 % of the bytes; each head of the
#      multi-head run equals its single-head run byte for byte; --heads with --fit-zarr writes the same fields as --fit-zarr alone
#  (1b) the GPU pyramid of the heads == the CPU pyramid at q 0 (shard 128 and 256)
#  (1c) the heads encoded from the GPU's levels (vcgpu.h) == the CPU encoder, q 8, every shard file of every level
#  (1d) the CT windows decoded on the GPU (UFSM_PRED_DEVREAD) == the host reader
#  (2) a plain latent net with the fibre head (--lat 4 --rv 2 --fib-head 1): --heads all writes rv, fib/H, fib/V, fib/dyad/c2..c7, fib/fam;
#      fib/H equals the old single-channel path (the pool4 chain in fp32) within one code on 1 % (the heads path rounds the upsampled logit
#      to fp16 as the slab path does: 0.64 % of the fixture's bytes land across a code boundary)
# Usage: test_predict_heads.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3)
P=(--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2 --q 0)
fail() { echo "FAIL: $*"; exit 1; }
cmpb() {   # cmpb A B MAXFRAC: two stores' 9.6 arrays over the fixture box: the fraction of bytes that differ, none by more than 1
  "$U" read "$1" 9.6 0 0 0 256 256 256 a.raw > /dev/null 2>&1 || fail "read $1"; "$U" read "$2" 9.6 0 0 0 256 256 256 b.raw > /dev/null 2>&1 || fail "read $2"
  python3 -I - a.raw b.raw "$3" "$1 vs $2" <<'PY'
import sys, numpy as np
a = np.fromfile(sys.argv[1], np.uint8).astype(int); b = np.fromfile(sys.argv[2], np.uint8).astype(int); d = np.abs(a - b)
print(f"  {sys.argv[4]}: {np.count_nonzero(d)} of {d.size} bytes differ, max {d.max()}, {np.count_nonzero(a)} set")
sys.exit(0 if d.max() <= 1 and np.count_nonzero(d) <= float(sys.argv[3]) * d.size and np.count_nonzero(a) > d.size // 8 else 1)
PY
}
# (1) sub-cell head, fine stem, rv 2, ink
"$U" train fx/sources-ink.json --out s "${C[@]}" --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --fstem 16 > s.log 2>&1 || fail "train (1) ($D/s.log)"
"$U" predict s/last.ckpt fx ct h "${P[@]}" --heads all > h.log 2>&1 || fail "predict --heads all ($D/h.log)"
grep -q 'predict: 2 heads from one trunk pass: rv (ch 1[0-9]+1[0-9] stem) ink (ch 1[0-9] stem)' h.log || fail "heads line ($D/h.log)"
for n in rv ink; do [ -e h/$n/9.6/zarr.json ] && [ -e h/$n/zarr.json ] || fail "store h/$n"; done
UFSM_PRED_HEADS=0 "$U" predict s/last.ckpt fx ct old "${P[@]}" --channel rv > old.log 2>&1 || fail "predict, old path ($D/old.log)"
cmpb old h/rv 0.001 || fail "rv: the heads path against the slab path"
"$U" predict s/last.ckpt fx ct one "${P[@]}" --channel rv > one.log 2>&1 || fail "predict --channel rv ($D/one.log)"
"$U" predict s/last.ckpt fx ct onei "${P[@]}" --channel ink > onei.log 2>&1 || fail "predict --channel ink ($D/onei.log)"
cmpb one h/rv 0 || fail "rv: single head against the multi-head run"
cmpb onei h/ink 0 || fail "ink: single head against the multi-head run"
"$U" predict s/last.ckpt fx ct f0 "${P[@]}" --fit-zarr f0z --no-out 1 > f0.log 2>&1 || fail "predict --fit-zarr ($D/f0.log)"
"$U" predict s/last.ckpt fx ct f1 "${P[@]}" --fit-zarr f1z --heads rv > f1.log 2>&1 || fail "predict --fit-zarr --heads rv ($D/f1.log)"
for a in pc ps sp t0 t3 w dz dq; do "$U" read f0z/$a "" 0 0 0 32 32 32 fa.raw > /dev/null 2>&1 && "$U" read f1z/$a "" 0 0 0 32 32 32 fb.raw > /dev/null 2>&1 || fail "read fit field $a"
                                   cmp -s fa.raw fb.raw || fail "fit field $a differs with --heads"; done
cmpb one f1/rv 0 || fail "rv beside --fit-zarr"
# the fit fields do not depend on the z-chunking of the slab path (a 1 MB planes' budget: 1-cell chunks with their winding halo planes)
UFSM_PRED_FIT_PLANES_MB=1 "$U" predict s/last.ckpt fx ct f2 "${P[@]}" --fit-zarr f2z --no-out 1 > f2.log 2>&1 || fail "predict --fit-zarr, small chunks ($D/f2.log)"
for a in pc ps sp t0 t3 w dz dq; do "$U" read f0z/$a "" 0 0 0 32 32 32 fa.raw > /dev/null 2>&1 && "$U" read f2z/$a "" 0 0 0 32 32 32 fb.raw > /dev/null 2>&1 || fail "read fit field $a"
                                   cmp -s fa.raw fb.raw || fail "fit field $a depends on the chunking"; done
# (1b) the GPU pyramid (each level-0 shard pooled on the device, levels >= 1 encoded from the pieces) against the CPU pyramid (each level
#      decoded back and pooled): at q 0 the same bytes, shard 128 (pieces below a chunk, gathered raw) and 256 (level 1 encoded at once)
for sh in 128 256; do
  for g in 1 0; do UFSM_PRED_GPU_PYR=$g "$U" predict s/last.ckpt fx ct py$sh$g --um 9.6 --level 0 --window 160 --halo 16 --threads 2 --q 0 --shard $sh --levels 3 --heads rv,ink > py$sh$g.log 2>&1 || fail "predict --levels 3 --shard $sh, GPU pyramid $g ($D/py$sh$g.log)"; done
  for n in rv ink; do for lv in "19.2 128" "38.4 64"; do set -- $lv
    "$U" read py${sh}1/$n $1 0 0 0 $2 $2 $2 a.raw > /dev/null 2>&1 && "$U" read py${sh}0/$n $1 0 0 0 $2 $2 $2 b.raw > /dev/null 2>&1 || fail "read the $n pyramid level $1 (shard $sh)"
    cmp -s a.raw b.raw || fail "$n level $1, shard $sh: the GPU pyramid differs from the CPU pyramid"
    [ "$(tr -d '\000' < a.raw | wc -c)" -gt $(($2 * $2 * $2 / 8)) ] || fail "$n level $1, shard $sh: (nearly) empty"; done; done
  grep -q 'pyramids pooled on the GPU' py${sh}1.log && ! grep -q 'pyramids pooled on the GPU' py${sh}0.log || fail "GPU pyramid switch (shard $sh)"
  grep -q '"path":"38.4"' py${sh}1/rv/zarr.json || fail "the GPU pyramid's group (shard $sh)"
done
# (1c) the GPU encoder's levels (vcgpu.h) at q 8: every shard file of every level the same bytes as the CPU encoder's
for sh in 128 256; do
  for g in 1 0; do UFSM_PRED_GPU_ENC=$g "$U" predict s/last.ckpt fx ct ge$sh$g --um 9.6 --level 0 --window 160 --halo 16 --threads 2 --q 8 --shard $sh --levels 3 --heads rv,ink > ge$sh$g.log 2>&1 || fail "predict --q 8 --shard $sh, GPU encoder $g ($D/ge$sh$g.log)"; done
  grep -q 'levels for the encoder from the GPU' ge${sh}1.log && ! grep -q 'levels for the encoder from the GPU' ge${sh}0.log || fail "GPU encoder switch (shard $sh)"
  [ "$(cd ge${sh}1 && find . -path '*/c/*' -type f | sort | xargs cat | md5sum)" = "$(cd ge${sh}0 && find . -path '*/c/*' -type f | sort | xargs cat | md5sum)" ] || fail "shard $sh: the GPU encoder's shards differ from the CPU encoder's"
  [ "$(cd ge${sh}1 && find . -path '*/c/*' -type f | wc -l)" -ge 6 ] || fail "shard $sh: too few shard files"
  UFSM_PRED_VCAP=10 UFSM_PRED_PROF=1 "$U" predict s/last.ckpt fx ct gv$sh --um 9.6 --level 0 --window 160 --halo 16 --threads 2 --q 8 --shard $sh --levels 3 --heads rv,ink > gv$sh.log 2>&1 || fail "predict, levels over capacity ($D/gv$sh.log)"
  grep -Eq 'head writers: .*, [1-9][0-9]* from their bytes' gv$sh.log || fail "shard $sh: the over-capacity fallback not taken ($D/gv$sh.log)"
  [ "$(cd gv$sh && find . -path '*/c/*' -type f | sort | xargs cat | md5sum)" = "$(cd ge${sh}0 && find . -path '*/c/*' -type f | sort | xargs cat | md5sum)" ] || fail "shard $sh: the over-capacity shards differ from the CPU encoder's"
done
# (1d) the CT windows decoded on the GPU into device memory (UFSM_PRED_DEVREAD): the same stores as the host reader's (windows over the
#      array's edges: fill), and the chunks went through the GPU
UFSM_PRED_DEVREAD=1 UFSM_PRED_PROF=1 "$U" predict s/last.ckpt fx ct dv "${P[@]}" --heads rv,ink > dv.log 2>&1 || fail "predict UFSM_PRED_DEVREAD ($D/dv.log)"
grep -q 'CT windows decoded into device memory' dv.log && grep -Eq 'device CT windows: [1-9][0-9]* chunks all on the GPU' dv.log || fail "device read not used ($D/dv.log)"
for n in rv ink; do cmpb h/$n dv/$n 0 || fail "$n: the device-read run differs from the host reader's"; done
UFSM_GPU_DECODE_ITEMS=4 UFSM_PRED_DEVREAD=1 UFSM_PRED_PROF=1 "$U" predict s/last.ckpt fx ct dvh "${P[@]}" --heads rv > dvh.log 2>&1 || fail "predict UFSM_PRED_DEVREAD, CPU fallbacks ($D/dvh.log)"
grep -Eq 'device CT windows: 0 chunks all on the GPU, [1-9][0-9]* with host substreams' dvh.log || fail "device read: the host fallback not taken ($D/dvh.log)"
cmpb h/rv dvh/rv 0 || fail "rv: the device-read run with CPU-decoded chunks differs"
# (2) plain latent net with the fibre head
"$U" train fx/sources-fibhv.json --out fb "${C[@]}" --lat 4 --rv 2 --rv-iso 100 --fib-head 1 > fb.log 2>&1 || fail "train (2) ($D/fb.log)"
"$U" predict fb/last.ckpt fx ct g "${P[@]}" --heads all > g.log 2>&1 || fail "predict --heads all, fibre ($D/g.log)"
for n in rv fib/H fib/V fib/dyad/c2 fib/dyad/c7 fib/fam; do [ -e g/$n/9.6/zarr.json ] || fail "store g/$n ($D/g.log)"; done
H=$(head -1 fb/last.ckpt | grep -o '"fib_dir":{"version":1,"head":1,"channels":{"H":[0-9]*' | grep -o '[0-9]*$')
UFSM_PRED_HEADS=0 "$U" predict fb/last.ckpt fx ct gh "${P[@]}" --channel $H > gh.log 2>&1 || fail "predict --channel $H, old path ($D/gh.log)"
cmpb gh g/fib/H 0.01 || fail "fib/H: the heads path against the pool4 path"
echo "test_predict_heads: ok"

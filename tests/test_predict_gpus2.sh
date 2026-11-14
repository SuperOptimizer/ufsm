#!/usr/bin/env bash
# test_predict_gpus2.sh UFSM MAKE_WIND_FIXTURE DIR: predict --gpus 0,1 on the latent paths (audit 11-14 infer #7: z-layer workers, one process
# per GPU): --heads all and --fit-zarr must give the bytes of the one-GPU run. With one visible GPU both workers share it (still two processes).
set -u
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
export UFSM_PRED_NO_AXIS=1   # (the fixture's predicts run without an axis, as test_predict_heads)
"$MF" fx 256 2 1 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
G=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$G" -ge 2 ] && GL=0,1 || GL=0,0
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 3)
P=(--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 2 --threads 2 --q 0 --box 0,0,0,256,256,256)
"$U" train fx/sources-ink.json --out s "${C[@]}" --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --fstem 16 > s.log 2>&1 || fail "train ($D/s.log)"
cmpraw() {   # cmpraw ROOT_A ROOT_B KEY N: the arrays' bytes over N^3
  "$U" read "$1" "$3" 0 0 0 $4 $4 $4 a.raw > /dev/null 2>&1 || fail "read $1/$3"; "$U" read "$2" "$3" 0 0 0 $4 $4 $4 b.raw > /dev/null 2>&1 || fail "read $2/$3"
  cmp -s a.raw b.raw || fail "$1/$3 differs from $2/$3"; }
"$U" predict s/last.ckpt fx ct h1 "${P[@]}" --heads all > h1.log 2>&1 || fail "predict --heads all, one GPU ($D/h1.log)"
"$U" predict s/last.ckpt fx ct h2 "${P[@]}" --heads all --gpus $GL > h2.log 2>&1 || fail "predict --heads all --gpus $GL ($D/h2.log)"
grep -q 'worker 1 of 2' h2.log || fail "no z-layer worker 1 ($D/h2.log)"
for n in rv ink; do cmpraw h1/$n h2/$n 9.6 256; cmpraw h1/$n h2/$n 19.2 128; [ -e h2/$n/zarr.json ] || fail "no group h2/$n"; done
[ -e h2/9.6/zarr.json ] && fail "the main store left beside the heads ($D/h2)"
"$U" predict s/last.ckpt fx ct f1 "${P[@]}" --fit-zarr f1z --fit-h 1 --no-out 1 > f1.log 2>&1 || fail "predict --fit-zarr, one GPU ($D/f1.log)"
"$U" predict s/last.ckpt fx ct f2 "${P[@]}" --fit-zarr f2z --fit-h 1 --no-out 1 --gpus $GL > f2.log 2>&1 || fail "predict --fit-zarr --gpus $GL ($D/f2.log)"
grep -Eq 'shard layers 1 \.\. 2 of 2' f2.log || fail "fields: worker 1 did not take the upper layer ($D/f2.log)"
[ -e f2z/fields.json ] && [ ! -e f2z/fields.json.w0 ] && [ ! -e f2z/fields.partial.json ] || fail "fields.json not finalised ($D/f2z)"
cmp -s f1z/fields.json f2z/fields.json || fail "fields.json differs"
for a in $(ls f1z); do [ -e f1z/$a/zarr.json ] || continue; cmpraw f1z/$a f2z/$a "" 256; done
"$U" predict s/last.ckpt fx ct g2 "${P[@]}" --fit-zarr g2z --fit-h 1 --heads rv --gpus $GL > g2.log 2>&1 || fail "predict --fit-zarr --heads rv --gpus ($D/g2.log)"
for a in pc ps w; do cmpraw f1z/$a g2z/$a "" 256; done
cmpraw h1/rv g2/rv 9.6 256
echo "test_predict_gpus2: ok"

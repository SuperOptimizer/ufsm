#!/usr/bin/env bash
# train --lat-dec (the learned sub-cell decoder, src/nn_latdec.cu) through the trainer's streamed head: a small latent net as the production
# ones (a fitted --lat-basis, --lat-guide 3, --fstem 16) trained 20 steps is the donor; --init-from it at lr 0 with and without --lat-dec 64
# must log the same losses (W2 starts at 0: the decoder adds nothing, the head's backward on the extra rows changes no weight at lr 0);
# trained at the normal lr the decoder must move the losses away from the plain head's; --lat-dec-raw 1 (64 voxels a cell) runs; --resume
# refuses a decoder mismatch. Usage: test_latdec_train.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 128 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000
   --fp4 2 --input-prec 8 --warmup 0 --wind4-tile 0 --seed 5)
"$U" train fx/sources.json --out fit "${C[@]}" --widths 32,64,128 --steps 20 --det 1 --lat 4 --lat-fit "$D/basis.lcb" --lat-fit-n 8 > fit.log 2>&1 || fail "lat-fit ($D/fit.log)"
L=(--widths 16,32,64 --lat 4 --lat-basis "$D/basis.lcb" --lat-c 29 --lat-step 0.25 --lat-guide 3 --fstem 16 --fstem-drop 0)
"$U" train fx/sources.json --out don "${C[@]}" "${L[@]}" --steps 20 > don.log 2>&1 || fail "donor ($D/don.log)"
"$U" train fx/sources.json --out a --det 1 --lr 0 --muon-lr 0 "${C[@]}" "${L[@]}" --steps 3 --init-from don/last.ckpt > a.log 2>&1 || fail "plain head at lr 0 ($D/a.log)"
"$U" train fx/sources.json --out b --det 1 --lr 0 --muon-lr 0 "${C[@]}" "${L[@]}" --steps 3 --init-from don/last.ckpt --lat-dec 64 > b.log 2>&1 || fail "decoder at lr 0 ($D/b.log)"
grep -q 'sub-cell decoder: 64 silu features' b.log || fail "no decoder line ($D/b.log)"
losses() { grep '^step ' "$1" | sed -E 's/[0-9.]+ ?(ms|s)\b//g; s/(lr|mlr|t|it\/s|win\/s|gn) [0-9.e+-]+//g' | cut -c1-200; }
la=$(losses a.log); lb=$(losses b.log)
[ -n "$la" ] && [ "$la" = "$lb" ] || { diff <(echo "$la") <(echo "$lb") | head; fail "the decoder at W2 = 0 changed the losses (lr 0)"; }
"$U" train fx/sources.json --out c --det 1 "${C[@]}" "${L[@]}" --steps 6 --init-from don/last.ckpt > c.log 2>&1 || fail "plain head trained ($D/c.log)"
"$U" train fx/sources.json --out e --det 1 "${C[@]}" "${L[@]}" --steps 6 --init-from don/last.ckpt --lat-dec 64 > e.log 2>&1 || fail "decoder trained ($D/e.log)"
[ "$(losses c.log | head -1)" = "$(losses e.log | head -1)" ] || fail "trained: step 1 must still match (W2 = 0 before the first update)"
[ "$(losses c.log | tail -1)" != "$(losses e.log | tail -1)" ] || fail "trained: the decoder did not change the losses by step 6 (no gradient reached W2?)"
grep -E '^step +6' e.log | grep -qiE 'nan|inf' && fail "non-finite losses with the decoder ($D/e.log)"
"$U" train fx/sources.json --out f --det 1 "${C[@]}" "${L[@]}" --steps 2 --init-from don/last.ckpt --lat-dec 32 --lat-dec-raw 1 > f.log 2>&1 || fail "raw decoder ($D/f.log)"
"$U" train fx/sources.json --out e --det 1 "${C[@]}" "${L[@]}" --steps 8 --resume e/last.ckpt --lat-dec 64 > e2.log 2>&1 || fail "resume with the decoder ($D/e2.log)"
! "$U" train fx/sources.json --out e --det 1 "${C[@]}" "${L[@]}" --steps 9 --resume e/last.ckpt > e3.log 2>&1 || fail "resume without --lat-dec accepted a decoder checkpoint"
# --lat-sub rv,ink (every sub-cell group: phase, rv pair, ink = 5 x 64 rows) with the raw decoder (no basis): W2 0 at lr 0 equal losses;
# predict --heads rv,ink of a random-W2 decoder (UFSM_LATDEC_TEST_W2) runs and differs from W2 0's (the decoder reaches the --lat-sub rows)
S=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 32,64,128 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --lat-sub rv,ink)
"$U" train fx/sources-ink.json --out sd "${S[@]}" --steps 10 > sd.log 2>&1 || fail "--lat-sub donor ($D/sd.log)"
"$U" train fx/sources-ink.json --out sa --lr 0 --muon-lr 0 "${S[@]}" --steps 2 --init-from sd/last.ckpt > sa.log 2>&1 || fail "--lat-sub at lr 0 ($D/sa.log)"
"$U" train fx/sources-ink.json --out sb --lr 0 --muon-lr 0 "${S[@]}" --steps 2 --init-from sd/last.ckpt --lat-dec 32 --lat-dec-raw 1 > sb.log 2>&1 || fail "--lat-sub + decoder at lr 0 ($D/sb.log)"
grep -q 'into the 320 sub-cell rows' sb.log || fail "the decoder does not cover the 5 --lat-sub groups ($D/sb.log)"
[ "$(losses sa.log)" = "$(losses sb.log)" ] || fail "--lat-sub: the decoder at W2 = 0 changed the losses"
UFSM_LATDEC_TEST_W2=2 "$U" train fx/sources-ink.json --out sc --lr 0 --muon-lr 0 "${S[@]}" --steps 1 --init-from sd/last.ckpt --lat-dec 32 --lat-dec-raw 1 > sc.log 2>&1 || fail "random W2 ($D/sc.log)"
P=(--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2 --q 0 --heads rv,ink --axis fx/axis.json)
"$U" predict sb/last.ckpt fx ct pb "${P[@]}" > pb.log 2>&1 || fail "predict W2 0 ($D/pb.log)"
"$U" predict sc/last.ckpt fx ct pc "${P[@]}" > pc.log 2>&1 || fail "predict random W2 ($D/pc.log)"
for h in rv ink; do "$U" read pb/$h 9.6 0 0 0 256 256 256 b.raw > /dev/null 2>&1 && "$U" read pc/$h 9.6 0 0 0 256 256 256 c.raw > /dev/null 2>&1 || fail "read $h"
  ! cmp -s b.raw c.raw || fail "predict --heads $h: the decoder's term does not reach it"; done
echo "test_latdec_train: ok"

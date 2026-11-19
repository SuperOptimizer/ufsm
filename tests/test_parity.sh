#!/usr/bin/env bash
# train-vs-predict parity (fix0332 2026-11-18 audit; the GN running-statistics bug was such a mismatch): a small latent net as the
# production ones (a fitted --lat-basis, --gn-ema unwarmed, --lat-guide 3, --fstem), one validation window through the training code (UFSM_PARITY_DUMP: its full-resolution
# logits and level-0 corner) and the same box through predict (window = the validation window, halo 0, --fit-zarr) must agree to the
# fp16 head rounding (max |dlogit| <= 0.02 every channel, >= 30 % of the values bit-equal), with the same numerics modes (weights hash,
# GN live, MX formats, latent step, FiLM codes) in both modes.txt lines.
# Usage: test_parity.sh UFSM MAKE_WIND_FIXTURE DIR [EXTRA TRAIN FLAGS] (e.g. "--lat-dec 32": the learned sub-cell decoder in both paths)
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3; X=(${4:-})
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
"$U" train fx/sources.json --out fit --P 128 --widths 32,64,128 --steps 20 --det 1 --task unwrap2 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 \
  --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --lat 4 --lat-fit "$D/basis.lcb" --lat-fit-n 8 > fit.log 2>&1 || fail "lat-fit ($D/fit.log)"
# (without a basis, --lat 4's cell-mean input, train and predict agree only to corr 0.85-0.97 on this fixture: DESIGN 2026-11-18 parity, open)
T=(--task unwrap2 --levels 1 --P 128 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8
   --warmup 0 --wind4-tile 0 --widths 16,32,64 --det 1 --lat 4 --lat-basis "$D/basis.lcb" --lat-c 29 --lat-step 0.25 --lat-guide 3 --fstem 16
   --gn-ema 1 --gn-ema-warm 1000000000 --seed 5 "${X[@]}")
"$U" train fx/sources.json --out a "${T[@]}" --steps 3 > a.log 2>&1 || fail "train ($D/a.log)"
mkdir -p tr pr
UFSM_PARITY_DUMP=$D/tr UFSM_PARITY_TAG=train UFSM_LAT_FUSE=0 "$U" train fx/sources.json --out b "${T[@]}" --steps 4 --resume a/last.ckpt --lr 0 --muon-lr 0 \
  --noaug 1 --val-batches 1 --val-every 1 --val-live 1 --val-store regen --val-train-ok 1 > b.log 2>&1 || fail "the validation pass ($D/b.log)"
[ -s tr/train.win ] && [ -s tr/train.logits ] || fail "no validation dump ($D/b.log)"
read -r _ _ _ lv _ cz cy cx _ _ _ P _ _ _ _ _ C < tr/train.win
[ "$lv" = 0 ] || fail "the validation window is at level $lv (the fixture trains level 0)"
UFSM_PARITY_DUMP=$D/pr UFSM_PARITY_TAG=predict "$U" predict a/last.ckpt fx ct pr/o --um 9.6 --level 0 --box $cz,$cy,$cx,$P,$P,$P --window $P --halo 0 \
  --grid-origin $cz,$cy,$cx --shard 128 --ema 0 --gpu 0 --threads 2 --axis fx/axis.json --levels 1 --fit-zarr pr/f --fit-h 2 --no-out 1 > p.log 2>&1 || fail "predict ($D/p.log)"
[ -s pr/predict.logits ] || fail "no predict dump ($D/p.log)"
grep -q 'DEFAULTED: predict --gn-live = 1' p.log || fail "predict did not default an unwarmed gn_ema checkpoint to live GN ($D/p.log)"
read -r _ _ _ _ _ PC _ < pr/predict.win
python3 - "$D" "$C" "$PC" "$P" <<'EOF' || fail "logits differ (above)"
import sys, numpy as np
d, ca, cb, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
A = np.fromfile(f'{d}/tr/train.logits', np.float16).reshape(ca, P, P, P); B = np.fromfile(f'{d}/pr/predict.logits', np.float16)
B = np.concatenate([B, np.zeros(cb * P**3 - B.size, np.float16)]).reshape(cb, P, P, P)
bad = 0
for c in range(min(ca, cb)):
    a, b = A[c].astype(np.float32), B[c].astype(np.float32)
    if not np.abs(a).max() > 0: continue   # (a channel the validation pass does not materialise)
    m = np.abs(a - b).max(); eq = np.mean(A[c].view(np.uint16) == B[c].view(np.uint16))
    print(f'test_parity: channel {c}: max |d| {m:.4f}, bit-equal {eq:.3f}, rms {np.sqrt((a * a).mean()):.3f}')
    bad += not (m <= 0.02 and eq >= 0.3)
sys.exit(bad != 0)
EOF
key() { grep "^$1 fwd_train 0" "$2" | head -1 | tr ' ' '\n' | grep -A1 -xE 'using_ema|gn_ema|ge_live|act_mx4|act_mx8|input_mx|input_prec|f16|half_res|lat_C|lat_step|lat_run_step|film|gfilm|w_hash' | grep -v -- '--' | paste -sd' '; }
a=$(key train tr/modes.txt); b=$(key predict pr/modes.txt)
[ -n "$a" ] && [ "$a" = "$b" ] || fail "numerics modes differ: train [$a] predict [$b]"
echo "test_parity: ok"

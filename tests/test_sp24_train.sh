#!/usr/bin/env bash
# sparse24 in training (the mid-run switch) on the wind fixture, a latent net with > 64-channel levels (the sparse fp4 kernel's range):
#  (1) --sparse24 3 --sparse24-mode 3 --sparse24-ramp 2: on at step 3, the mask fully applied at step 4, finite losses, the sparse kernel ran,
#      after every hard step the master keeps all its energy under its own mask (pruned weights exactly 0, UFSM_SP24_CHECK), header keys
#  (2) a resume without the sparse flags stays sparse (mode, hard) and pruned
#  (3) predict on the sparse checkpoint runs the sparse kernel, and its output equals the dense kernels' on the same masked weights (UFSM_SP24=0)
#  (4) mode 2 with a ramp across a resume (the ramp recomputed from the flags): on, mask applied at its step, finite
# Usage: test_sp24_train.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --warmup 0 --seed 5 --wind4-tile 0 --widths 96,128,160 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --fstem 16 --opt muon)
UFSM_SP24_REPORT=1 UFSM_SP24_CHECK=1 "$U" train fx/sources-ink.json --out s "${C[@]}" --steps 8 --sparse24 3 --sparse24-mode 3 --sparse24-ramp 2 > s.log 2>&1 \
  || fail "train (1) ($D/s.log)"
grep -q '^step 3: 2:4 sparsity on (mode 3, levels all, ramp 2 steps, hard' s.log || fail "switch line ($D/s.log)"
grep -q '^step 4: sparse24 mask applied; kept |w|^2 per level: L0 ' s.log || fail "mask applied at step 4 ($D/s.log)"
! grep -q 'non-finite' s.log || fail "non-finite step ($D/s.log)"
n=$(grep -c '^sparse24 check: master kept |w|^2 1.000000' s.log); [ "$n" = 5 ] || fail "pruned master after steps 4..8: $n of 5 ($D/s.log)"
grep -Eq '^sparse24: [1-9][0-9]* convs on the sparse fp4 kernel' s.log || fail "sparse kernel did not run ($D/s.log)"
head -1 s/last.ckpt | grep -q '"sparse24":3' && head -1 s/last.ckpt | grep -q '"sparse24_hard":1' && ! head -1 s/last.ckpt | grep -q '"sparse24_r"' || fail "header keys"
UFSM_SP24_REPORT=1 UFSM_SP24_CHECK=1 "$U" train fx/sources-ink.json --out s "${C[@]}" --steps 10 --resume s/last.ckpt > r.log 2>&1 || fail "resume (2) ($D/r.log)"
n=$(grep -c '^sparse24 check: master kept |w|^2 1.000000' r.log); [ "$n" = 2 ] || fail "resumed: pruned master $n of 2 ($D/r.log)"
grep -Eq '^sparse24: [1-9][0-9]* convs on the sparse fp4 kernel' r.log && ! grep -q 'non-finite' r.log || fail "resumed run ($D/r.log)"
P=(--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2 --q 0)
UFSM_SP24_REPORT=1 "$U" predict s/last.ckpt fx ct ps "${P[@]}" --channel rv > ps.log 2>&1 || fail "predict ($D/ps.log)"
grep -Eq '^sparse24: [1-9][0-9]* convs on the sparse fp4 kernel' ps.log || fail "predict did not run the sparse kernel ($D/ps.log)"
UFSM_SP24=0 UFSM_SP24_REPORT=1 "$U" predict s/last.ckpt fx ct pd "${P[@]}" --channel rv > pd.log 2>&1 || fail "predict, dense kernels ($D/pd.log)"
grep -q '^sparse24: 0 convs on the sparse fp4 kernel' pd.log || fail "UFSM_SP24=0 still sparse ($D/pd.log)"
"$U" read ps 9.6 0 0 0 256 256 256 a.raw > /dev/null 2>&1 && "$U" read pd 9.6 0 0 0 256 256 256 b.raw > /dev/null 2>&1 || fail "read predictions"
cmp -s a.raw b.raw || fail "predict: sparse kernel output != dense kernels on the masked weights"
python3 -I -c "import sys,numpy as np; a=np.fromfile('a.raw',np.uint8); sys.exit(0 if np.count_nonzero(a) > a.size // 8 else 1)" || fail "prediction mostly empty"
"$U" train fx/sources-ink.json --out m "${C[@]}" --steps 4 --sparse24 3 --sparse24-mode 2 --sparse24-ramp 3 > m.log 2>&1 || fail "mode 2 ramp ($D/m.log)"
head -1 m/last.ckpt | grep -q '"sparse24_r":0.333' || fail "mid-ramp key (r at step 4: 1 - 2 / 3)"
"$U" train fx/sources-ink.json --out m "${C[@]}" --steps 7 --sparse24 3 --sparse24-mode 2 --sparse24-ramp 3 --resume m/last.ckpt > m2.log 2>&1 || fail "mode 2 resume ($D/m2.log)"
grep -q '^step 5: sparse24 mask applied' m2.log && ! grep -q 'non-finite' m2.log || fail "mode 2: mask after the resumed ramp ($D/m2.log)"
echo "test_sp24_train: ok"

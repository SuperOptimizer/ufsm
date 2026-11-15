#!/usr/bin/env bash
# predict's run guards (unroll audit 11-14):
#  #16 the non-finite guard on the fit-field accumulators: a clean --fit-zarr run reports nothing; with a NaN planted in them
#      (UFSM_TEST_NAN_LOGITS=1) the run reports the slabs and fails (exit 1), also on the host --fit-fields path;
#  #17 --resume: the store's attrs carry the run (checkpoint sha256, window, halo, q, ...); a resume of the same run goes on (and skips
#      the listed shards), one with another q / window / checkpoint, or over a store written before the run record, is refused (exit 2).
# Usage: test_predict_guard.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
T=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 2 --lat 4)
"$U" train fx/sources.json --out a "${T[@]}" --seed 5 > a.log 2>&1 || fail "train ($D/a.log)"
"$U" train fx/sources.json --out b "${T[@]}" --seed 6 > b.log 2>&1 || fail "train b ($D/b.log)"
P=(--um 9.6 --level 0 --gpu 0 --threads 2 --shard 128 --levels 1 --full-grid 1)

# ---- #16 ---- (the guard counts the fit-field accumulators: a NaN logit there would become undefined field bytes)
F=(--um 9.6 --level 0 --gpu 0 --threads 2 --shard 128 --levels 1 --box 0,0,0,128,128,128 --window 128 --halo 16)
"$U" predict a/last.ckpt fx ct p0 "${F[@]}" --fit-zarr p0/fz > p0.log 2>&1 || fail "predict --fit-zarr ($D/p0.log)"
grep -q 'FAILED' p0.log && fail "a clean run reported non-finite values ($D/p0.log)"
set +e; UFSM_TEST_NAN_LOGITS=1 "$U" predict a/last.ckpt fx ct pn "${F[@]}" --fit-zarr pn/fz > pn.log 2>&1; rc=$?; set -e
[ "$rc" = 1 ] || fail "a NaN run exited $rc, not 1 ($D/pn.log)"
grep -Eq 'FAILED: [1-9][0-9]* fit-field slabs held non-finite values' pn.log || fail "no guard report ($D/pn.log)"
set +e; UFSM_TEST_NAN_LOGITS=1 "$U" predict a/last.ckpt fx ct ph "${F[@]}" --fit-fields ph/ff > ph.log 2>&1; rc=$?; set -e
[ "$rc" = 1 ] || fail "a NaN run (host --fit-fields) exited $rc, not 1 ($D/ph.log)"

# ---- #17 ----
R=(--resume 1 --window 128 --halo 16 --q 0)
"$U" predict a/last.ckpt fx ct r "${P[@]}" "${R[@]}" --box 0,0,0,128,128,128 > r1.log 2>&1 || fail "resume run 1 ($D/r1.log)"
DONE=$(ls r/predict-done-*.txt); [ -s "$DONE" ] || fail "empty done list"
ZJ=$(ls r/*/zarr.json | head -1)
grep -q "\"checkpoint_sha256\":\"$(sha256sum a/last.ckpt | cut -c1-64)\"" "$ZJ" || fail "the store has no checkpoint sha256 ($ZJ)"
grep -q '"run":{.*"window":128,"halo":16,.*"q":0,' "$ZJ" || fail "the run record lacks window / halo / q ($ZJ)"
"$U" predict a/last.ckpt fx ct r "${P[@]}" "${R[@]}" --box 0,0,0,128,256,128 > r2.log 2>&1 || fail "the same run resumed ($D/r2.log)"
grep -Eq '[1-9][0-9]* already written \(--resume\)' r2.log || fail "the resume skipped nothing ($D/r2.log)"
cp "$ZJ" zarr.json.keep
refuse() {   # refuse LOG ARGS...: predict must exit 2 with the refusal
  local log=$1; shift; set +e; "$U" predict "$@" > "$log" 2>&1; local rc=$?; set -e
  [ "$rc" = 2 ] || fail "resume over another run exited $rc, not 2 ($D/$log)"
  grep -q 'refusing to mix them' "$log" || fail "no refusal message ($D/$log)"
  cmp -s "$ZJ" zarr.json.keep || fail "the refused run changed the store's zarr.json ($D/$log)"
}
refuse rq.log a/last.ckpt fx ct r "${P[@]}" --resume 1 --window 128 --halo 16 --q 8 --box 0,0,0,128,128,128
refuse rw.log a/last.ckpt fx ct r "${P[@]}" --resume 1 --window 160 --halo 16 --q 0 --box 0,0,0,128,128,128
refuse rc.log b/last.ckpt fx ct r "${P[@]}" "${R[@]}" --box 0,0,0,128,128,128
sed -E 's/,"run":\{[^}]*\}//' zarr.json.keep > "$ZJ"; cp "$ZJ" zarr.json.keep
refuse ro.log a/last.ckpt fx ct r "${P[@]}" "${R[@]}" --box 0,0,0,128,128,128
grep -q 'an unrecorded (older binary) run' ro.log || fail "the old-store message ($D/ro.log)"
echo "test_predict_guard: ok"

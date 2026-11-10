#!/bin/bash
# predict --latents end to end on the synthetic spiral fixture: a cell transform fitted (train --lat-fit), two 6-step --lat-basis
# --rv nets (plain head; guided head --lat-guide 1), the CT encoded with ufsm lc-encode (whole volume; and a box at an origin off the
# shard grid), then predict --channel rv from the CT and from the store. The outputs differ only by the store's quantiser (the
# kernels' half-step bound: test_lat), so the check is a bound on the disagreement, plus: the cell-grid input and the full-resolution
# input give the same bytes; no CT voxel is read (store io of the CT root: metadata only); windows off the store's cell grid are refused.
# Usage: test_latents.sh <ufsm binary> <make_wind_fixture binary>
set -u
U=$1 F=$2
D=$(mktemp -d /tmp/test_latents_XXXXXX)
fail() { echo "FAIL: $*"; echo "kept $D"; exit 1; }
"$F" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1 || fail "fixture"
T="--task unwrap2 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --lat 4"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/fit" --P 128 --widths 32,64,128 --steps 20 $T --lat-fit "$D/basis.lcb" --lat-fit-n 8 > "$D/fit.log" 2>&1 || fail "lat-fit ($D/fit.log)"
for g in 0 1; do
    # shellcheck disable=SC2086
    "$U" train "$D/fx/sources.json" --out "$D/n$g" --P 64 --widths 32,64,128 --steps 6 $T --lat-basis "$D/basis.lcb" --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 \
        --rv 1 --rv-iso 64 --lat-guide $g > "$D/n$g.log" 2>&1 || fail "train lat-guide $g ($D/n$g.log)"
done
"$U" lc-encode "$D/basis.lcb" "$D/fx" ct/9.6 "$D/lc" --shard 16 --chunk 8 --threads 4 > "$D/lc.log" 2>&1 || fail "lc-encode ($D/lc.log)"
"$U" lc-encode "$D/basis.lcb" "$D/fx" ct/9.6 "$D/lcb" --shard 16 --chunk 8 --threads 4 --box 32,0,0,192,256,256 > "$D/lcb.log" 2>&1 || fail "lc-encode --box ($D/lcb.log)"
P="--um 9.6 --channel rv --axis $D/fx/axis.json --levels 1 --q 0"
run() {   # TAG NET STORE|- [args]: predict, then the box as raw bytes
    local tag=$1 net=$2 st=$3; shift 3; local x=(); [ "$st" != - ] && x=(--latents "$st")
    # shellcheck disable=SC2086
    UFSM_PRED_IO=1 "$U" predict "$D/$net/last.ckpt" "$D/fx" ct "$D/p_$tag" $P "$@" "${x[@]}" > "$D/p_$tag.log" 2>&1 || return 1
    "$U" read "$D/p_$tag" 9.6 0 0 0 256 256 256 "$D/p_$tag.raw" > /dev/null 2>&1
}
run g0ct n0 - --window 128 --halo 16 || fail "predict g0 CT"
run g0lc n0 "$D/lc" --window 128 --halo 16 || fail "predict g0 latents ($D/p_g0lc.log)"
UFSM_PRED_LAT_INPUT=0 run g0lf n0 "$D/lc" --window 128 --halo 16 || fail "predict g0 latents, full-resolution input"
run g1ct n1 - --window 288 --halo 16 --shard 128 || fail "predict g1 CT"
run g1lc n1 "$D/lc" --window 288 --halo 16 --shard 128 || fail "predict g1 latents ($D/p_g1lc.log)"
run bct n0 - --window 128 --halo 16 --box 32,0,0,192,256,256 || fail "predict box CT"
run blc n0 "$D/lcb" --window 128 --halo 16 --box 32,0,0,192,256,256 || fail "predict box latents ($D/p_blc.log)"
run off n0 "$D/lc" --window 128 --halo 18 && fail "a window off the cell grid was accepted"
grep -q "off the store's 4-voxel cell grid" "$D/p_off.log" || fail "no off-grid message ($D/p_off.log)"
cmp -s "$D/p_g0lc.raw" "$D/p_g0lf.raw" || fail "--latents: the cell-grid input and the full-resolution input differ"
grep "^store io:.*$D/fx\$" "$D/p_g0lc.log" | awk '{ if ($3 + 0 > 0.05) exit 1 }' || fail "--latents read CT voxels: $(grep '^store io' "$D/p_g0lc.log")"
grep -q "^predict io: the latent store" "$D/p_g0lc.log" || fail "no latent store read accounting"
python3 -I - "$D" <<'EOF' || fail "outputs ($D)"
import sys, numpy as np
d = sys.argv[1]
def ld(t): return np.fromfile(f"{d}/p_{t}.raw", np.uint8).astype(int).reshape(256, 256, 256)
bad = 0
for a, b, agree_min, box in (("g0ct", "g0lc", 0.97, None), ("g1ct", "g1lc", 0.80, None), ("bct", "blc", 0.97, (32, 224))):
    x, y = ld(a), ld(b)
    if box: x, y = x[box[0]:box[1]], y[box[0]:box[1]]
    m = (x > 0) & (y > 0)
    nzx, nzy = (x > 0).mean(), (y > 0).mean()
    agree = ((x[m] > 127) == (y[m] > 127)).mean(); md = np.abs(x - y)[m].mean()
    ok = agree >= agree_min and md <= 16 and abs(nzx - nzy) < 0.03 and m.mean() > 0.5
    print(f"{a} / {b}: written {nzx:.4f} / {nzy:.4f}, side agreement {agree:.4f} (>= {agree_min}), mean |d| {md:.2f} (<= 16): {'ok' if ok else 'FAIL'}")
    bad += not ok
sys.exit(1 if bad else 0)
EOF
rm -rf "$D"
echo "test_latents: ok"

#!/bin/bash
# train --lat-direct-p (sample.h lat_direct_p, train.c lat_hook): direct windows take the trunk's input from their stored latvol cells.
#  1. --lat-direct-p 0 with a "latvol" twin in the sources = no twin: the same losses byte for byte (--det 1, lr 0): the option is inert.
#  2. --lat-direct-p 1: every new window is direct (the lat-direct line counts them; replays are voxel windows), finite losses.
#  3. the same windows with the hook off (UFSM_LAT_DIRECT_OFF=1: their input from the rebuilt CT by the CT -> latent kernel, the trainer's
#     own quantiser): the losses at lr 0 within a few percent (the two quantisers differ by at most half a step; a wrong z-score, scale or
#     cell placement would move them far more).
#  4. a store with another transform is refused (its cells mean another input).
# Usage: test_lat_direct.sh <ufsm binary> <make_wind_fixture binary>
set -u
U=$1 F=$2
D=$(mktemp -d /tmp/test_lat_direct_XXXXXX)
fail() { echo "FAIL: $*"; echo "kept $D"; exit 1; }
"$F" "$D/fx" 256 2 1 > "$D/fx.log" 2>&1 || fail "fixture"
T="--task unwrap2 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --lat 4"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/fit" --P 128 --widths 32,64,128 --steps 20 $T --lat-fit "$D/basis.lcb" --lat-fit-n 8 > "$D/fit.log" 2>&1 || fail "lat-fit ($D/fit.log)"
# shellcheck disable=SC2086
"$U" train "$D/fx/sources.json" --out "$D/fit2" --P 128 --widths 32,64,128 --steps 20 $T --lat-fit "$D/basis2.lcb" --lat-fit-n 4 --seed 9 > "$D/fit2.log" 2>&1 || fail "lat-fit 2 ($D/fit2.log)"
mkdir -p "$D/lv" "$D/lv2"
"$U" latvol "$D/basis.lcb" "$D/fx" ct "$D/lv/ct" --levels 9.6 --threads 4 --gpu 0 > "$D/lv.log" 2>&1 || fail "latvol ($D/lv.log)"
"$U" latvol "$D/basis2.lcb" "$D/fx" ct "$D/lv2/ct" --levels 9.6 --threads 4 --gpu 0 > "$D/lv2.log" 2>&1 || fail "latvol 2 ($D/lv2.log)"
python3 -I - "$D" <<'EOF' || fail "sources json"
import json, sys
d = sys.argv[1]; j = json.load(open(d + '/fx/sources.json'))
for name, root in (('lv', d + '/lv'), ('lv2', d + '/lv2')):
    for s in j['sources'] if isinstance(j, dict) else j: s['latvol'] = {'root': root, 'ct': s.get('ct', 'ct')}
    json.dump(j, open('%s/sources-%s.json' % (d, name), 'w'))
EOF
L="--lat-basis $D/basis.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lr 0 --muon-lr 0 --det 1 --mix-every 3"
run() {   # run NAME JSON [flags...]
  local n=$1 j=$2; shift 2
  # shellcheck disable=SC2086
  "$U" train "$j" --out "$D/$n" --P 64 --widths 32,64,128 --steps 6 "$@" $T $L > "$D/$n.log" 2>&1 || fail "train $n ($D/$n.log)"
}
losses() { grep -o '^step *[0-9]* .* loss [0-9.]*' "$D/$1.log" | grep -o 'loss [0-9.]*' | tr '\n' ' '; }
run a "$D/fx/sources.json"
run b "$D/sources-lv.json" --lat-direct-p 0
[ "$(losses a)" = "$(losses b)" ] && [ -n "$(losses a)" ] || fail "--lat-direct-p 0 with a latvol twin changed the losses: $(losses a) | $(losses b)"
run c "$D/sources-lv.json" --lat-direct-p 1
grep -q 'have a "latvol" twin' "$D/c.log" || fail "no --lat-direct-p startup line ($D/c.log)"
read -r built used <<< "$(grep -o 'lat-direct: [0-9]* direct windows built, [0-9]* trained' "$D/c.log" | tail -1 | awk '{print $2, $6}')"
[ "${built:-0}" -gt 0 ] && [ "${used:-0}" -gt 0 ] || fail "no direct windows built / trained ($D/c.log)"
grep -qiE 'nan|inf ' <(losses c) && fail "non-finite losses with direct windows: $(losses c)"
UFSM_LAT_DIRECT_OFF=1 run d "$D/sources-lv.json" --lat-direct-p 1
python3 -I - "$(losses c)" "$(losses d)" <<'EOF' || fail "direct vs rebuilt-CT input: losses too far apart: $(losses c) | $(losses d)"
import sys
c = [float(x) for x in sys.argv[1].split()[1::2]]; d = [float(x) for x in sys.argv[2].split()[1::2]]
assert c and len(c) == len(d), (c, d)
worst = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(c, d)); print('direct vs rebuilt-CT input: worst relative loss difference %.4f' % worst)
assert worst < 0.05
EOF
# shellcheck disable=SC2086
"$U" train "$D/sources-lv2.json" --out "$D/e" --P 64 --widths 32,64,128 --steps 2 --lat-direct-p 1 $T $L > "$D/e.log" 2>&1 && fail "a store with another transform was accepted ($D/e.log)"
grep -q "not the network's" "$D/e.log" || fail "no transform-mismatch message ($D/e.log)"
echo "test_lat_direct: OK (p 0 inert; $built direct windows, $used trained from cells; direct vs rebuilt-CT losses close; another transform refused)"
rm -rf "$D"

#!/usr/bin/env bash
# train --init-from maps an unwrap2 donor's heads by name (2026-11-12: u1 -> U2 copied the head by position, so U2's rv pair got u1's
# verso row and an untrained pad row; rv bce 0.57 at step 50 against u1's 0.36). A donor with --rv 2 (rv 11-12) initialises a
# --ink-head 1 --rv 2 net (ink 11, rv 12-13): at lr 0 its step-1 rv bce matches the donor layout's own (within 0.03; --det 1: the batch order
# independent of thread timing), and the log names the mapping. Usage: test_init_heads.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
C=(--task unwrap2 --rv 2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000
   --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64)
"$U" train fx/sources.json --out don "${C[@]}" --steps 40 > don.log 2>&1 || fail "donor ($D/don.log)"
"$U" train fx/sources.json --out a --det 1 --lr 0 --muon-lr 0 "${C[@]}" --steps 2 --init-from don/last.ckpt > a.log 2>&1 || fail "same layout ($D/a.log)"
"$U" train fx/sources.json --out b --det 1 --lr 0 --muon-lr 0 --ink-head 1 "${C[@]}" --steps 2 --init-from don/last.ckpt > b.log 2>&1 || fail "ink + rv ($D/b.log)"
grep -q 'init-from: unwrap2 donor by head: rows 0-10 kept, rv 12-13 <- donor 11-12, ink 11 <- fresh' b.log || fail "no by-head mapping line ($D/b.log)"
ra=$(grep '^step      1' a.log | grep -o 'rv bce [0-9.]*' | awk '{print $3}'); rb=$(grep '^step      1' b.log | grep -o 'rv bce [0-9.]*' | awk '{print $3}')
python3 -c "import sys; a, b = float('$ra'), float('$rb'); sys.exit(0 if abs(a - b) < 0.03 else 1)" || fail "rv bce after the mapping $rb vs the donor layout's $ra"
# the second case (test_fibhead_train caught it): a by-head init that also adds the fibre head must leave the fibre rows fresh (they were
# random): ink + rv + fibre from the rv donor (by head) against rv + fibre from it (the positional fibre branch), lr 0, the same windows
"$MF" fxf 256 2 1 > fxf.log 2>&1
"$U" train fxf/sources-fibhv.json --out c --det 1 --lr 0 --muon-lr 0 --ink-head 1 --fib-head 1 "${C[@]}" --steps 2 --init-from don/last.ckpt > c.log 2>&1 || fail "ink + rv + fibre ($D/c.log)"
"$U" train fxf/sources-fibhv.json --out d --det 1 --lr 0 --muon-lr 0 --fib-head 1 "${C[@]}" --steps 2 --init-from don/last.ckpt > d.log 2>&1 || fail "rv + fibre ($D/d.log)"
grep -q 'init-from: unwrap2 donor by head: .*fibre 1[0-9]-2[0-9] fresh' c.log || fail "no fresh-fibre mapping line ($D/c.log)"
grep -q 'init-from: head rows 0-12 copied, rows 13-21 (fibre head) fresh' d.log || fail "the positional fibre branch's line ($D/d.log)"
fc=$(grep '^step      1' c.log | grep -o 'fib bce [0-9.]*' | awk '{print $3}'); fd=$(grep '^step      1' d.log | grep -o 'fib bce [0-9.]*' | awk '{print $3}')
[ -n "$fc" ] && [ "$fc" = "$fd" ] || fail "fibre bce at step 1: by head $fc vs the fibre branch $fd (fresh rows in both)"
# the third case (2026-11-19): a plain --init-from of a sub-cell (--lat-guide 3) donor keeps its sub-cell rows; the "start at zero" reset ran
# after every such copy, so try9 warm starts began on the interpolated head (phase error 81-89 deg at step 50 against try9's 54)
L=(--lat 4 --lat-guide 3)
"$U" train fx/sources.json --out sd "${C[@]}" "${L[@]}" --steps 40 > sd.log 2>&1 || fail "sub-cell donor ($D/sd.log)"
"$U" train fx/sources.json --out se --det 1 --lr 0 --muon-lr 0 "${C[@]}" "${L[@]}" --steps 2 --init-from sd/last.ckpt > se.log 2>&1 || fail "sub-cell init ($D/se.log)"
grep -q 'sub-cell head: outputs [0-9]*-[0-9]* from the donor' se.log || fail "sub-cell rows not kept from the donor ($D/se.log)"
! grep -q 'sub-cell head: .* start at zero' se.log || fail "sub-cell rows reset after the copy ($D/se.log)"
"$U" train fx/sources.json --out sg "${C[@]}" --lat 4 --steps 2 > sg.log 2>&1 || fail "plain latent donor ($D/sg.log)"
"$U" train fx/sources.json --out sf --det 1 --lr 0 --muon-lr 0 "${C[@]}" "${L[@]}" --steps 2 --init-from sg/last.ckpt > sf.log 2>&1 || fail "plain latent donor into a sub-cell net ($D/sf.log)"
grep -q 'sub-cell head: .* start at zero' sf.log || fail "a donor without sub-cell rows must leave them at zero ($D/sf.log)"
echo "test_init_heads: ok (rv bce at step 1: donor layout $ra, ink + rv mapped by head $rb; fresh fibre rows: fib bce $fc in both paths)"

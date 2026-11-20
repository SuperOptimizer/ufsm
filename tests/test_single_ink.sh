#!/usr/bin/env bash
# train --single ink (2026-11-20): a single-target net, the head the ink row only (unet cout 1), the unwrap2 frame's other heads, losses
# and device label planes off. On the wind fixture's ink target:
#  1. an unwrap2 --ink-head 1 donor; --single ink --init-from it maps the donor's ink row into row 0, and at lr 0 on the same windows
#     (--det 1) its ink bce equals the donor layout's at every step (the same trunk, the same ink row);
#  2. the three level-0 buffers under head slabs (unet.l3h: a 16-wide level 0, dec[1] 16): the planner takes them, and at lr 0 the ink
#     losses equal the four-buffer layout's (UFSM_L0_3BUF=0);
#  3. an 8-wide level 0 trains (three buffers only); training steps run with finite ink losses in both layouts (>= 3 of 6 windows with ink), the checkpoint header is unwrap version 3 (single ink, ink 0), the net
#     has one output, and a resume continues.
# Usage: test_single_ink.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
C=(--task unwrap2 --ink-head 1 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1
   --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --dec-widths 0,16)
inkl() { grep '^step' "$1" | grep -o 'ink2* bce [0-9.na]*' | awk '{print $3}'; }
"$U" train fx/sources-ink.json --out don "${C[@]}" --steps 20 > don.log 2>&1 || fail "donor ($D/don.log)"
"$U" train fx/sources-ink.json --out a --det 1 --lr 0 --muon-lr 0 "${C[@]}" --steps 3 --init-from don/last.ckpt > a.log 2>&1 || fail "donor layout at lr 0 ($D/a.log)"
"$U" train fx/sources-ink.json --out b --det 1 --lr 0 --muon-lr 0 "${C[@]}" --single ink --steps 3 --init-from don/last.ckpt > b.log 2>&1 || fail "single at lr 0 ($D/b.log)"
grep -q "init-from: --single ink: head row 0 <- the donor's ink row" b.log || fail "no ink row mapping line ($D/b.log)"
grep -q 'shared encoder a1, three level-0 buffers, ' b.log || fail "single: the planner did not take the three level-0 buffers ($D/b.log)"
ia=$(inkl a.log); ib=$(inkl b.log)
[ -n "$ia" ] && [ "$(echo "$ia" | wc -l)" = 3 ] || fail "no ink losses in the donor layout ($D/a.log)"
paste <(echo "$ia") <(echo "$ib") | awk '{d = $1 - $2; if (d < 0) d = -d; if (!($2 == $2) || d > 1e-4 + 1e-3 * $1) bad = 1} END {exit bad}' \
  || { paste <(echo "$ia") <(echo "$ib"); fail "single vs the donor layout: ink bce at lr 0"; }
UFSM_L0_3BUF=0 "$U" train fx/sources-ink.json --out c --det 1 --lr 0 --muon-lr 0 "${C[@]}" --single ink --steps 3 --init-from don/last.ckpt > c.log 2>&1 || fail "single, four buffers ($D/c.log)"
! grep -q 'shared encoder a1, three level-0 buffers, ' c.log || fail "UFSM_L0_3BUF=0 still took the three buffers ($D/c.log)"
ic=$(inkl c.log)
[ "$ib" = "$ic" ] || { paste <(echo "$ib") <(echo "$ic"); fail "three vs four level-0 buffers: ink bce at lr 0"; }
for m in t3 t4; do
  E=(); [ $m = t4 ] && E=(env UFSM_L0_3BUF=0)
  "${E[@]}" "$U" train fx/sources-ink.json --out $m "${C[@]}" --single ink --steps 6 --init-from don/last.ckpt > $m.log 2>&1 || fail "single training ($D/$m.log)"
  it=$(inkl $m.log | tr '\n' ' '); echo "$it" | awk '{n = 0; for (i = 1; i <= NF; i++) { if ($i != $i + 0) exit 1; n += $i > 0 } exit !(NF == 6 && n >= 3)}' \
    || fail "$m: ink bce over 6 steps: '$it' (the fixture's windows without ink: 0)"
done
# an 8-wide level 0 (the head's backward only through the three level-0 buffers' GN-recomputing kernel)
C8=("${C[@]/#16,32,64/8,16,32}"); C8=("${C8[@]/#0,16/0,8}")   # (opt takes the first --widths: replace, do not append)
"$U" train fx/sources-ink.json --out w8 "${C8[@]}" --single ink --steps 4 > w8.log 2>&1 || fail "8-wide level 0 ($D/w8.log)"
grep -q 'shared encoder a1, three level-0 buffers, ' w8.log && grep -q '^step      4 ' w8.log || fail "8-wide level 0: no three-buffer plan or no step 4 ($D/w8.log)"
head -1 t3/last.ckpt | grep -q '"unwrap":{"version":3,"single":"ink","ink":0' || fail "header: no unwrap version 3 single ink"
"$U" train fx/sources-ink.json --out t3 "${C[@]}" --single ink --steps 8 --resume t3/last.ckpt > r.log 2>&1 || fail "resume ($D/r.log)"
grep -q '^step      8 ' r.log || fail "resume: no step 8 ($D/r.log)"
echo "test_single_ink: ok (ink bce at lr 0, donor layout / single three buffers / single four: $(echo $ia | tr ' ' ,) / $(echo $ib | tr ' ' ,) / $(echo $ic | tr ' ' ,))"

#!/usr/bin/env bash
# train --single fib (2026-11-20): a single-target fibre net, the head the fibre head's 9 rows only (H / V masks, the axis dyad, its
# family; unet cout 9), the unwrap2 frame's other heads, losses and device label planes off. On the wind fixture's tube rasters:
#  1. an unwrap2 --fib-head 1 donor; --single fib --init-from it maps the donor's fibre rows (its "fib_dir" channels) into rows 0-8, and
#     at lr 0 on the same windows (--det 1) its fibre losses equal the donor layout's at every step;
#  2. the three level-0 buffers under head slabs with 9 logit channels (the GN-recomputing head backward, CI 9): the planner takes them,
#     and at lr 0 the losses equal the four-buffer layout's (UFSM_L0_3BUF=0);
#  3. training steps run with finite fibre losses; the header is unwrap version 3 (single fib) with the fib_dir map at 0-8; a resume continues.
# Usage: test_single_fib.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
fail() { echo "FAIL: $*"; exit 1; }
C=(--task unwrap2 --fib-head 1 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1
   --ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --dec-widths 0,16)
fibl() { grep '^step' "$1" | grep -o 'fib bce [0-9.na]* dir [0-9.na]*' | awk '{print $3, $5}'; }
"$U" train fx/sources-fibhv.json --out don "${C[@]}" --steps 20 > don.log 2>&1 || fail "donor ($D/don.log)"
"$U" train fx/sources-fibhv.json --out a --det 1 --lr 0 --muon-lr 0 "${C[@]}" --steps 3 --init-from don/last.ckpt > a.log 2>&1 || fail "donor layout at lr 0 ($D/a.log)"
"$U" train fx/sources-fibhv.json --out b --det 1 --lr 0 --muon-lr 0 "${C[@]}" --single fib --steps 3 --init-from don/last.ckpt > b.log 2>&1 || fail "single at lr 0 ($D/b.log)"
grep -q "init-from: --single fib: head rows 0-8 <- 11,12,13,14,15,16,17,18,19 (the donor's fibre rows)" b.log || fail "no fibre row mapping line ($D/b.log)"
grep -q 'shared encoder a1, three level-0 buffers, ' b.log || fail "single: the planner did not take the three level-0 buffers ($D/b.log)"
fa=$(fibl a.log); fb=$(fibl b.log)
[ -n "$fa" ] && [ "$(echo "$fa" | wc -l)" = 3 ] || fail "no fibre losses in the donor layout ($D/a.log)"
paste <(echo "$fa") <(echo "$fb") | awk '{for (i = 1; i <= 2; i++) { d = $i - $(i + 2); if (d < 0) d = -d; if (!($(i + 2) == $(i + 2)) || d > 1e-4 + 1e-3 * $i) bad = 1 }} END {exit bad}' \
  || { paste <(echo "$fa") <(echo "$fb"); fail "single vs the donor layout: fibre losses at lr 0"; }
UFSM_L0_3BUF=0 "$U" train fx/sources-fibhv.json --out c --det 1 --lr 0 --muon-lr 0 "${C[@]}" --single fib --steps 3 --init-from don/last.ckpt > c.log 2>&1 || fail "single, four buffers ($D/c.log)"
! grep -q 'shared encoder a1, three level-0 buffers, ' c.log || fail "UFSM_L0_3BUF=0 still took the three buffers ($D/c.log)"
fc=$(fibl c.log)
[ "$fb" = "$fc" ] || { paste <(echo "$fb") <(echo "$fc"); fail "three vs four level-0 buffers: fibre losses at lr 0"; }
"$U" train fx/sources-fibhv.json --out t "${C[@]}" --single fib --steps 6 > t.log 2>&1 || fail "single fresh training ($D/t.log)"
ft=$(fibl t.log | tr '\n' ' '); echo "$ft" | awk '{for (i = 1; i <= NF; i++) if ($i != $i + 0) exit 1; exit !(NF == 12)}' || fail "fresh: fibre losses over 6 steps: '$ft'"
grep -q 'init-from' t.log && fail "fresh run mentions init-from ($D/t.log)"
head -1 t/last.ckpt | grep -q '"unwrap":{"version":3,"single":"fib"}' || fail "header: no unwrap version 3 single fib"
head -1 t/last.ckpt | grep -q '"fib_dir":{"version":1,"head":1,"channels":{"H":0,"V":1,"dyad":\[2,3,4,5,6,7\],"family_v":8}' || fail "header: fib_dir map"
"$U" train fx/sources-fibhv.json --out t "${C[@]}" --single fib --steps 8 --resume t/last.ckpt > r.log 2>&1 || fail "resume ($D/r.log)"
grep -q '^step      8 .* fib bce' r.log || fail "resume: no step 8 ($D/r.log)"
echo "test_single_fib: ok (fib bce / dir at lr 0, donor layout / single three buffers / single four: $(echo $fa | tr ' ' ,) / $(echo $fb | tr ' ' ,) / $(echo $fc | tr ' ' ,))"

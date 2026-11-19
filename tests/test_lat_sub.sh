#!/usr/bin/env bash
# train --lat-sub (2026-11-19): sub-cell rows for the rv pair and the ink output of a latent-grid net (--lat 4 --lat-guide 3), after the
# phase pair's. (1) --init-from a phase-only donor grows the head (its rows by position, the new sub-cell rows zero): at lr 0 the step-1
# losses equal the donor layout's byte for byte (a zero sub-cell term adds nothing), the header names the layout ("lat_sub":[0,1,12,13,11]);
# (2) predict --heads rv,ink of that grown checkpoint == the donor layout's (the shifted sub-cell base read right); (3) after training the new
# rows, predict --heads rv differs (the sub-cell terms reach predict) and the losses still finish.
# Usage: test_lat_sub.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
rd() { "$U" read "$1" 9.6 0 0 0 256 256 256 "$2" > /dev/null 2>&1 || fail "read $1"; }   # (the stores' 9.6 arrays: zarr.json names the checkpoint)
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 32,64,128 --det 1 --lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1)   # (level-0 width 32: a head over 192 outputs backward on the tensor cores only)
P=(--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2 --q 0 --heads rv,ink --axis fx/axis.json)
"$U" train fx/sources-ink.json --out don "${C[@]}" --steps 20 > don.log 2>&1 || fail "donor ($D/don.log)"
"$U" train fx/sources-ink.json --out a --lr 0 --muon-lr 0 "${C[@]}" --steps 2 --init-from don/last.ckpt > a.log 2>&1 || fail "donor layout ($D/a.log)"
"$U" train fx/sources-ink.json --out b --lr 0 --muon-lr 0 "${C[@]}" --steps 2 --init-from don/last.ckpt --lat-sub rv,ink > b.log 2>&1 || fail "--lat-sub ($D/b.log)"
grep -q 'init-from: --lat-sub: head rows 0-[0-9]* by position, the new sub-cell rows' b.log || fail "no growth line ($D/b.log)"
head -c 4096 b/last.ckpt | tr -d '\0' | grep -q '"lat_sub":\[0,1,12,13,11\]' || fail "header lat_sub ($D/b/last.ckpt)"
la=$(grep '^step      1 ' a.log | sed 's/ qk .*//; s/ gn .*//'); lb=$(grep '^step      1 ' b.log | sed 's/ qk .*//; s/ gn .*//')
# (every number within 2e-4: the wider head's forward rounds a few wind4 logits one fp16 code apart, 1 in the 4th decimal of its worst gauge)
[ -n "$la" ] && paste <(tr ' ' '\n' <<< "$la") <(tr ' ' '\n' <<< "$lb") | awk '{a = $1; b = $2; gsub(/[^0-9.e+-]/, "", a); gsub(/[^0-9.e+-]/, "", b)
  if (a ~ /^[0-9.]+$/ && b ~ /^[0-9.]+$/) { if ((a - b) > 2e-4 || (b - a) > 2e-4) bad = 1 } else if ($1 != $2) bad = 1 } END { exit bad }' \
  || fail "step-1 losses with zero new rows: [$lb] vs the donor layout [$la]"
"$U" predict a/last.ckpt fx ct pa "${P[@]}" > pa.log 2>&1 || fail "predict a ($D/pa.log)"
"$U" predict b/last.ckpt fx ct pb "${P[@]}" > pb.log 2>&1 || fail "predict b ($D/pb.log)"
for h in rv ink; do rd pa/$h a.raw; rd pb/$h b.raw; cmp -s a.raw b.raw || fail "predict --heads $h: grown head (zero rows) != donor layout"; done
"$U" train fx/sources-ink.json --out c "${C[@]}" --steps 30 --init-from don/last.ckpt --lat-sub rv,ink > c.log 2>&1 || fail "--lat-sub training ($D/c.log)"
grep -q '^step     30 ' c.log || fail "no step 30 ($D/c.log)"
"$U" predict c/last.ckpt fx ct pc "${P[@]}" > pc.log 2>&1 || fail "predict c ($D/pc.log)"
grep -q 'rv (ch 12+13 sub-cell' pc.log || fail "predict does not read rv's sub-cell rows ($D/pc.log)"
rd pb/rv b.raw; rd pc/rv c.raw; ! cmp -s b.raw c.raw || fail "trained sub-cell rows left predict --heads rv unchanged"
echo "test_lat_sub: ok (step-1 losses equal with zero new rows; --heads rv / ink equal; trained rows reach predict)"

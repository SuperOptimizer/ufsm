#!/usr/bin/env bash
# predict --ct-shell + ufsm ct-shell-list (inkeval 2026-11-27, the lossless / q2 CT shell for 3-D ink on coarse scans):
#  1. ct-shell-list over a box + --pad: x runs of 128^3 chunks, header "q 0" (auto: lossless under --shell-max-gb); with a tiny limit
#     "q 2" and "DEFAULTED: ctshell q2 (lossless est ..)";
#  2. a lossless shell (latvol --volcomp 0 --exact 1 of the same level) read by predict --ct-shell: "from the shell", no DEFAULTED, the
#     prediction byte-identical to the main store's;
#  3. a q8 shell: read (the prediction differs from the main store's);
#  4. a shell that misses chunks in the windows' reach (--pad 0, a larger predict box): DEFAULTED, the main store, identical output.
# Usage: test_predict_shell.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1
T=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 2 --lat 4)
"$U" train fx/sources.json --out a "${T[@]}" --seed 5 > a.log 2>&1 || fail "train ($D/a.log)"
P=(--um 9.6 --level 0 --gpu 0 --threads 2 --shard 128 --levels 1 --full-grid 1 --window 128 --halo 16 --q 0 --no-axis 1)
python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<II', 0x424c4655, 1) + bytes(4 * (64 + 64 * 64)))" > id.lcb
same() { diff -rq --exclude='*.json' --exclude='*.txt' "$1" "$2" > /dev/null; }

# 1
"$U" ct-shell-list fx ct all.boxes --level 0 --um 9.6 --box 0,0,0,128,128,128 --pad 1024 > l1.log 2>&1 || fail "ct-shell-list ($D/l1.log)"
head -1 all.boxes | grep -q '^# level 0 shard 1024 q 0 ' || fail "header not q 0: $(head -1 all.boxes)"
grep -q DEFAULTED l1.log && fail "a small shell DEFAULTED ($D/l1.log)"
awk '!/^#/ && ($1 % 128 || $2 % 128 || $3 % 128 || $4 != 128 || $5 != 128 || $6 % 128) {exit 1}' all.boxes || fail "a run off the chunk grid"
"$U" ct-shell-list fx ct q2.boxes --level 0 --um 9.6 --box 0,0,0,128,128,128 --pad 1024 --shell-max-gb 0.001 > l2.log 2>&1 || fail "ct-shell-list q2 ($D/l2.log)"
head -1 q2.boxes | grep -q '^# level 0 shard 1024 q 2 ' || fail "a shell over the limit not q 2: $(head -1 q2.boxes)"
grep -Eq '^DEFAULTED: ctshell q2 \(lossless est [0-9.e+-]+ GB > 0.001\)' l2.log || fail "no DEFAULTED for the q2 pick ($D/l2.log)"

# 2 and 3
for q in 0 8; do
  "$U" latvol id.lcb fx ct s$q/ct --levels 9.6 --volcomp $q --gpu -1 --threads 2 --exact 1 --boxes all.boxes > lv$q.log 2>&1 || fail "latvol q$q ($D/lv$q.log)"
  cp all.boxes s$q/ct/shell.boxes
done
"$U" predict a/last.ckpt fx ct pm "${P[@]}" --box 0,0,0,128,128,128 > pm.log 2>&1 || fail "predict main ($D/pm.log)"
"$U" predict a/last.ckpt fx ct p0 "${P[@]}" --box 0,0,0,128,128,128 --ct-shell s0 > p0.log 2>&1 || fail "predict lossless shell ($D/p0.log)"
grep -q 'from the shell' p0.log && ! grep -q DEFAULTED p0.log || fail "the lossless shell was not read ($D/p0.log)"
same pm p0 || fail "the lossless shell's prediction differs from the main store's"
"$U" predict a/last.ckpt fx ct p8 "${P[@]}" --box 0,0,0,128,128,128 --ct-shell s8 > p8.log 2>&1 || fail "predict q8 shell ($D/p8.log)"
grep -q 'from the shell' p8.log || fail "the q8 shell was not read ($D/p8.log)"
same pm p8 && fail "the q8 shell's prediction equals the main store's (was the shell read?)"

# 4
"$U" ct-shell-list fx ct part.boxes --level 0 --um 9.6 --box 0,0,0,128,128,128 --pad 0 > l3.log 2>&1 || fail "ct-shell-list --pad 0 ($D/l3.log)"
"$U" latvol id.lcb fx ct sp/ct --levels 9.6 --volcomp 8 --gpu -1 --threads 2 --exact 1 --boxes part.boxes > lvp.log 2>&1 || fail "latvol part ($D/lvp.log)"
cp part.boxes sp/ct/shell.boxes
"$U" predict a/last.ckpt fx ct pb "${P[@]}" --box 0,0,0,256,256,256 > pb.log 2>&1 || fail "predict main, big box ($D/pb.log)"
"$U" predict a/last.ckpt fx ct pp "${P[@]}" --box 0,0,0,256,256,256 --ct-shell sp > pp.log 2>&1 || fail "predict partial shell ($D/pp.log)"
grep -Eq 'DEFAULTED: predict: --ct-shell sp: not used \([0-9]+ of the [0-9]+ chunks' pp.log || fail "a partial shell was not refused loudly ($D/pp.log)"
same pb pp || fail "the refused shell changed the prediction"
echo "test_predict_shell: OK"

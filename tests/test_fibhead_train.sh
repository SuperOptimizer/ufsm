#!/usr/bin/env bash
# train --fib-head (the unwrap2 fibre head: H / V tube masks from the sources' "fibre_h" / "fibre_v" targets + the fib_dir direction) on the
# wind fixture's tube rasters: whole-voxel trunk; latent (--lat 4 --rv 2) with the label stream check (the fibre planes' +-r halo planes,
# 0 differing bytes) and streamed == device-kept planes; --init-from a run without the head (rows kept, fibre rows fresh) and a resume;
# the fine stem with the head (16 stem outputs, the pad channels excluded); a fibre-only source (fibre_h / fibre_v, no band / phase) draws around papyrus cells in the stored target chunks, alone and mixed
# with a winding source
# Usage: test_fibhead_train.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
. "$(dirname "$(readlink -f "$0")")/det_cmp.sh"
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
C=(--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1)
L=(--rv 2 --rv-iso 100 --lat 4)
ok() { grep -q "$2" "$1" || { echo "FAIL: $3 ($D/$1)"; exit 1; }; }
"$U" train fx/sources-fibhv.json --out v "${C[@]}" --steps 4 --fib-head 1 > v.log 2>&1 || { echo "FAIL: voxel run ($D/v.log)"; exit 1; }
ok v.log '^step      4 .* fib bce 0\.[0-9]* dir 0\.[0-9]* (100% of windows)' "voxel run: fibre losses"
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 UFSM_LAT_STREAM=check "$U" train fx/sources-fibhv.json --out ls "${C[@]}" "${L[@]}" --steps 4 --fib-head 1 > ls.log 2>&1 || { echo "FAIL: latent run ($D/ls.log)"; exit 1; }
[ "$(grep -c 'lat stream check: .* 0 of [0-9]* slab bytes differ' ls.log)" -gt 0 ] && ! grep 'lat stream check' ls.log | grep -vq ' 0 of ' || { echo "FAIL: stream check"; grep 'lat stream check' ls.log | tail -2; exit 1; }
UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 UFSM_LAT_LDEV=0 "$U" train fx/sources-fibhv.json --out ld "${C[@]}" "${L[@]}" --steps 4 --fib-head 1 > ld.log 2>&1 || { echo "FAIL: latent run, streamed ($D/ld.log)"; exit 1; }
for f in ls ld; do grep '^step' $f.log | sed -E 's/ [0-9.]+ samp\/s.*//' > $f.txt; done
det_cmp ls.log ld.log "device-kept planes vs streamed" || exit 1
# the fine stem with the fibre head (the head padded past 16 outputs: the stem maps into the 16 supervised voxel outputs, not the pad)
"$U" train fx/sources-fibhv.json --out fs "${C[@]}" "${L[@]}" --lat-guide 3 --fstem 16 --ink-head 1 --steps 4 --fib-head 1 > fs.log 2>&1 || { echo "FAIL: latent run with --fstem ($D/fs.log)"; exit 1; }
ok fs.log 'fine stem: 16 silu features of 4^3 voxels at stride 2 into 16 head outputs' "fstem + fib-head: 16 stem outputs"
ok fs.log '^step      4 .* fib bce 0\.[0-9]* dir 0\.[0-9]*' "fstem + fib-head: fibre losses"
"$U" train fx/sources-fibhv.json --out don "${C[@]}" "${L[@]}" --steps 2 > don.log 2>&1 || { echo "FAIL: donor ($D/don.log)"; exit 1; }
"$U" train fx/sources-fibhv.json --out ini "${C[@]}" "${L[@]}" --steps 3 --fib-head 1 --init-from don/last.ckpt > ini.log 2>&1 || { echo "FAIL: init-from ($D/ini.log)"; exit 1; }
ok ini.log 'init-from: head rows 0-12 copied, rows 13-21 (fibre head) fresh' "init-from rows"
"$U" train fx/sources-fibhv.json --out ini "${C[@]}" "${L[@]}" --steps 5 --fib-head 1 --resume ini/last.ckpt > res.log 2>&1 || { echo "FAIL: resume ($D/res.log)"; exit 1; }
ok res.log '^step      5 .* fib bce' "resume"
head -1 ini/last.ckpt | grep -q '"fib_dir":{"version":1,"head":1,"channels":{"H":13,"V":14' || { echo "FAIL: header fib_dir"; exit 1; }
python3 - <<'PY'
import json
a = json.load(open('fx/sources-fibhv.json'))['sources'][0]; f = json.loads(json.dumps(a)); f['targets'].pop('band'); f['name'] = 'fib-only'
w = json.load(open('fx/sources.json'))['sources'][0]; f['weight'] = 0.25
json.dump({'sources': [f]}, open('fx/sources-fibonly.json', 'w')); json.dump({'sources': [w, f]}, open('fx/sources-fibmix.json', 'w'))
PY
for s in fibonly fibmix; do
  "$U" train fx/sources-$s.json --out $s "${C[@]}" --steps 6 --fib-head 1 > $s.log 2>&1 || { echo "FAIL: $s run ($D/$s.log)"; exit 1; }
  ok $s.log 'sampler: fib-only: [0-9]* of [0-9]* level-[0-9] CT cells hold papyrus inside the stored target chunks' "$s: fibre-only draw cells"
  ok $s.log '^step      6 .* fib bce 0\.[0-9]*' "$s: fibre losses"
done
grep '^step' fibmix.log | grep -Eq 'phase bce ([1-9]|0\.[1-9])' || { echo "FAIL: fibmix: no winding windows"; exit 1; }
echo "test_fibhead_train: ok"

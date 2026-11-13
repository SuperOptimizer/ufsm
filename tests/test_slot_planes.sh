#!/usr/bin/env bash
# the sampler slots' host planes (2026-11-13, hostmem): the plane bank (sample.h plane_bank: a slot holds as many label planes as the
# neediest source fills, the all-unknown label roles of a window on one shared 255 plane) and slot_skip 4 (the unpacked targets t / m
# dropped where the trainer never reads them: --lat --split z). Both must leave every batch plane and every device input byte-identical:
# --det 1 training on a winding source + a fibre-only source (side / w4 / wcomp real on the first, the fibre planes on the second), the
# old layout (UFSM_PLANE_BANK=0, UFSM_TM_SKIP4=0) against the new one with UFSM_BANK_CHECK=1 (aborts if the 255 plane was written):
# whole windows (--gpu-aug 3: w4 runs, ink bits) and the latent split on one device (--gpus 0,0 --split z: --gpu-aug falls back to 1,
# unpacked targets, every label plane a byte plane).
# Usage: test_slot_planes.sh UFSM MAKE_WIND_FIXTURE DIR
set -eu
. "$(dirname "$(readlink -f "$0")")/det_cmp.sh"
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); D=$3
rm -rf "$D"; mkdir -p "$D"; cd "$D"
"$MF" fx 256 2 1 > fx.log 2>&1
python3 - <<'PY'
import json
a = json.load(open('fx/sources-fibhv.json'))['sources'][0]; f = json.loads(json.dumps(a)); f['targets'].pop('band'); f['name'] = 'fib-only'
w = json.load(open('fx/sources.json'))['sources'][0]; f['weight'] = 1.0
json.dump({'sources': [w, f]}, open('fx/sources-fibmix.json', 'w'))
PY
C=(--task unwrap2 --levels 1 --P 64 --B 1 --workers 2 --sampler-slots 4 --val-batches 0 --val-every 1000 --log-every 1 --ckpt-every 1000 --fp4 2
   --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 8 --fib-head 1 --rv 2 --rv-iso 100 --lat 4 --gpu-aug 3)
run() {   # NAME ENV... -- ARGS...
  local n=$1; shift; local e=(); while [ "$1" != -- ]; do e+=("$1"); shift; done; shift
  env UFSM_BATCH_HASH=1 UFSM_DET_TRACE=1 "${e[@]}" "$U" train fx/sources-fibmix.json --out $n "${C[@]}" "$@" > $n.log 2>&1 || { echo "FAIL: run $n ($D/$n.log)"; tail -3 $n.log; exit 1; }
}
for mode in whole split; do
  X=(--gpus 0); [ $mode = split ] && X=(--gpus 0,0 --split z)
  run ${mode}_old UFSM_PLANE_BANK=0 UFSM_TM_SKIP4=0 -- "${X[@]}"
  run ${mode}_new UFSM_BANK_CHECK=1 UFSM_SPLIT_LWARP=0 -- "${X[@]}"   # (split: the host label warps of --gpu-aug 1, as the old layout)
  grep -q 'sampler: [0-9] label planes per slot for [0-9] label roles' ${mode}_new.log || { echo "FAIL: $mode: the plane bank is off ($D/${mode}_new.log)"; exit 1; }
  ! grep -q 'label planes per slot' ${mode}_old.log || { echo "FAIL: $mode: UFSM_PLANE_BANK=0 kept the bank"; exit 1; }
  grep -q 'fib bce 0\.[0-9]*' ${mode}_new.log && grep -Eq 'phase bce ([1-9]|0\.[1-9])' ${mode}_new.log || { echo "FAIL: $mode: no fibre or no winding windows"; exit 1; }
  if [ $mode = whole ]; then det_cmp ${mode}_old.log ${mode}_new.log "$mode: the old slot layout vs the plane bank" || exit 1
  else   # (the split steps print no dettrace lines: the batch planes exact, the losses within det_cmp's 2%)
    ha=$(grep '^batchhash' split_old.log); hb=$(grep '^batchhash' split_new.log)
    [ -n "$ha" ] && [ "$ha" = "$hb" ] || { echo "FAIL: split: the batches differ"; diff <(echo "$ha") <(echo "$hb") | head -4; exit 1; }
    paste -d' ' <(grep '^step' split_old.log | awk '{print $2, $6}') <(grep '^step' split_new.log | awk '{print $2, $6}') | awk '{d = $2 - $4; if (d < 0) d = -d; n++; if ($1 != $3 || d > 0.02 * ($2 < 0 ? -$2 : $2) + 1e-6) bad = 1} END {exit bad || n < 8}' \
      || { echo "FAIL: split: losses differ / missing steps"; exit 1; }
    DET_CMP_N=$(echo "$ha" | wc -l)
  fi
  echo "$mode: $(grep -o 'sampler: [0-9] label planes per slot for [0-9] label roles' ${mode}_new.log), $DET_CMP_N batches identical"
done
# the latent split with --gpu-aug 2 / 3 on unpacked targets (the default since 2026-11-13: the labels warped on the device by z range,
# w4 runs, ink bits; equal to the host warp within test_gpu_aug's 1e-4 of the voxels, so: the CT bytes and records exact, losses within 2%)
run split_dev UFSM_BANK_CHECK=1 -- --gpus 0,0 --split z
grep -q -- '--gpu-aug 2 without packed targets' split_dev.log && grep -q 'winding mod 4 as runs' split_dev.log || { echo "FAIL: split: no --gpu-aug 2 / w4 runs ($D/split_dev.log)"; exit 1; }
for f in ct8 aug rad; do
  [ "$(grep -o "^batchhash [0-9]* slot [0-9]* $f [0-9a-f]*" split_new.log | sed 's/ slot [0-9]*//')" = "$(grep -o "^batchhash [0-9]* slot [0-9]* $f [0-9a-f]*" split_dev.log | sed 's/ slot [0-9]*//')" ] || { echo "FAIL: split: $f differs with the device warps"; exit 1; }
done
paste -d' ' <(grep '^step' split_new.log | awk '{print $2, $6}') <(grep '^step' split_dev.log | awk '{print $2, $6}') | awk '{d = $2 - $4; if (d < 0) d = -d; n++; if ($1 != $3 || d > 0.02 * ($2 < 0 ? -$2 : $2) + 1e-6) bad = 1} END {exit bad || n < 8}' \
  || { echo "FAIL: split: device-warped losses differ by > 2%"; paste <(grep '^step' split_new.log | awk '{print $2, $6}') <(grep '^step' split_dev.log | awk '{print $2, $6}'); exit 1; }
echo "split: --gpu-aug 2 on the device: $(grep -o 'sampler: [0-9] label planes per slot for [0-9] label roles' split_dev.log), CT bytes exact, losses within 2%"
grep -q 'targets: packed' split_new.log && { echo "FAIL: split: packed targets (the case slot_skip 4 is for)"; exit 1; }
echo "test_slot_planes: ok"

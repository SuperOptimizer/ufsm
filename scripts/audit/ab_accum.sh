#!/bin/bash
# Audit 2026-11-14 train #8: --accum 5 vs --accum 2 at equal windows (prepared, not yet run).
# Arm A: accum 5 at try6's lr (Muon 0.003 / AdamW 3e-4). Arm B: accum 2 at lr x 0.77 (0.00231 / 2.31e-4; ~sqrt(2/5) x 1.22, the audit's
# choice). Both: --init-from the same try6 checkpoint, --det 1 and one --seed (the same window stream: equal windows = the same data,
# A makes 2/5 as many updates), the scratch2 net single card at P 512 on the wind fixture (the pod's only legal data), try6's XFLAGS
# (--gn-ema-warm 1e9 etc.), WSD warmup 50 windows, cooldown 0.2, --opt-stats.
# Usage (from the dir holding fx/ and klt1.lcb; one card, under the pod lock):
#   flock /root/gpu.lock timeout -k 10 9000 scripts/audit/ab_accum.sh BIN OUTDIR [WINDOWS=2000] [SEED=7] [INIT=/root/admuon/run/try6.ckpt]
# Then: scripts/audit/ab_accum.sh --summary OUTDIR  (loss / phase / wind4 / acc and, when the log has it, the "(kl ...)" group, at equal
# windows: the last 20 % (the cooldown) and windows 1000-1400; paired per log line, the same windows in both arms).
set -u
if [ "${1:-}" = --summary ]; then
  D=$2
  for a in A_accum5 B_accum2; do [ -f "$D/r_$a.log" ] || { echo "no $D/r_$a.log"; exit 1; }; done
  awk -v last=${3:-0} '
    function grab(s, key,   m) { if (match(s, key " -?[0-9.]+")) { m = substr(s, RSTART, RLENGTH); sub(/.* /, "", m); return m + 0 } return "" }
    function klgrp(s,   g) { if (!match(s, /\(kl [^;)]*/)) return ""; g = substr(s, RSTART + 4, RLENGTH - 4); return " " g }   # (the "(kl phase X rv X normal X wind4 X ...;" group; values may be slightly negative)
    FNR == 1 { arm = (FILENAME ~ /A_accum5/) ? "A" : "B" }
    /^step +[0-9]+ lr/ { st = $2 + 0; if (st > mx) mx = st
      split("loss|phase bce|wind4 bce| acc| phase| normal| wind4|phase error", K, "|"); kg = klgrp($0)
      for (k = 1; k <= 8; k++) { v = (k >= 5 && k <= 7) ? (kg == "" ? "" : grab(kg, K[k])) : grab($0, K[k]); if (v != "") { V[arm, st, k] = v; have[k] = 1 } } S[st] = 1 }
    END {
      split("loss|phase|wind4|acc|kl phase|kl normal|kl wind4|phase err deg", N, "|")
      for (r = 1; r <= 2; r++) {
        lo = r == 1 ? int(mx * 0.8) + 1 : 1001; hi = r == 1 ? mx : 1400
        printf "windows %d-%d:", lo, hi
        for (k = 1; k <= 8; k++) { if (!have[k]) continue; sa = sb = n = 0
          for (st in S) if (st + 0 >= lo && st + 0 <= hi && ((("A", st, k) in V) && (("B", st, k) in V))) { sa += V["A", st, k]; sb += V["B", st, k]; n++ }
          if (n) printf "  %s A %.4f B %.4f (B-A %+.4f, n %d)", N[k], sa / n, sb / n, (sb - sa) / n, n }
        printf "\n" } }' "$D/r_A_accum5.log" "$D/r_B_accum2.log"
  exit 0
fi
BIN=${1:-} OUT=${2:-}
[ -n "$BIN" ] && [ -n "$OUT" ] || { echo "usage: $0 BIN OUTDIR [WINDOWS] [SEED] [INIT]"; exit 2; }
WIN=${3:-2000} SEED=${4:-7} INIT=${5:-/root/admuon/run/try6.ckpt}
mkdir -p "$OUT"
for arm in "A_accum5 5 0.003 0.0003" "B_accum2 2 0.00231 0.000231"; do
  read -r NAME ACC MLR ALR <<< "$arm"
  R="$OUT/r_$NAME"; case "$R" in "$OUT"/r_*) rm -rf -- "$R" ;; esac
  df -h . | tail -n 1
  env UFSM_HEAD_SLAB_Z=16 CUDA_VISIBLE_DEVICES=${GPU:-0} "$BIN" train fx/sources-ink.json --out "$R" --gpus 0 --P ${P:-512} --B 1 \
    --widths 256,320,384,448,288,288 --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1 --fibre-gpu 0 --ink-gpu 0 \
    --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --normal-step 1 \
    --lat 4 --lat-basis ${LATB:-klt1.lcb} --lat-c 29 --lat-step 0.25 --lat-guide 3 --film 2 --fstem 16 --fstem-drop 0.25 --uw 0 \
    --ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1 --zfix 1 --aug none \
    --steps "$WIN" --sched wsd --cooldown 0.2 --warmup 50 --opt muon --muon-lr $MLR --lr $ALR --muon-tc 2 \
    --fp4 2 --gpu-aug 3 --mem auto --workers 6 --sampler-slots 4 --log-every 10 --opt-stats 50 --val-batches 0 --val-every 1000000 \
    --gn-ema-warm 1000000000 --skip-empty 1 --skip-thr 0 --skip-margin 1 --accum $ACC \
    --ckpt-every 1000000 --seed "$SEED" --det 1 --init-from "$INIT" > "$R.log" 2>&1
  echo "arm $NAME accum $ACC lr $MLR / $ALR rc $?" >> "$R.log"
  case "$R" in "$OUT"/r_*) rm -rf -- "$R" ;; esac   # (the 2 GB final checkpoint: the pod's disk is nearly full; the logs are the result)
done
"$0" --summary "$OUT"

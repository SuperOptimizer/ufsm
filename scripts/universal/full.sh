#!/usr/bin/env bash
# scripts/universal/full.sh RUN_DIR: the full-dataset universal run on one desk GPU (GPU, default 0) under launch.sh's watchdog
#   sources: mksources.sh = the real GT (labelled-desk.json: Paris 4 L2, 0139, 0814; holdouts) + the wind96 pseudo-label boxes of all scans
#            (PSEUDO, default /vesuvius/ufsm/dino/pseudo/scans: copied there by scripts/dino/pseudo-sync.sh; pseudo-sources.sh drops boxes
#            under 2 % labelled or touching a holdout; group weight = PW x the GT's) + every local U0 scan as SSL-only (ct_only, scanprep
#            masks); PHerc1667 excluded; the Paris 4 78 keV / 0139 / 0814 / 1447 eval boxes held out
#   recipe:  the laptop gate-arm recipe (iBOT on the attention tokens over a 16-sample SSL batch + GT SupCon on the main window, --gt 1)
#   window:  P (default 512) when a 30-step bench at it runs (16 GB GPU), else 384; source window 1.25 P (zoom 0.7-1.25), LP = locals
# Stop: touch RUN_DIR/STOP. Resume: rerun (launch.sh resumes from RUN_DIR/last.ckpt).
set -u
R=${1:?run dir}; mkdir -p $R; R=$(cd $R && pwd); D=$(cd "$(dirname "$0")" && pwd)
U=${U:-/vesuvius/ufsm/dino/src/build/ufsm}; GPU=${GPU:-0}
export PSEUDO=${PSEUDO:-/vesuvius/ufsm/dino/pseudo/scans} PW=${PW:-1} MINPCT=${MINPCT:-2}
cfg() {   # $1 = P
  echo "--recipe 1 --P $1 --source-P $(( ($1 * 5 / 4 + 63) / 64 * 64 )) --zoom ${ZOOM:-0.7,1.25} --local-P ${LP:-64} --locals 1 --attn-blocks 2 --attn-reg 1 --cls reg
    --exclude-src PHerc1667 --unlabelled-share ${US:-0.5} --gt ${GT:-1} --object 0 --region 0 --ink 0 --dino 0 --ibot 0 --koleo 0 --var 0 --proto-lr 0
    --dim 32 --head-init 0.1 --warmup ${WARM:-200} --widths 16,64,128,288 --head-slab 16 --mem lean --linear-mx 4 --protos-k 1024 --ibot-k 1024
    --mask-input 1 --ssl-workers 8 --ssl-ibot 1 --ssl-ibot-src tok --ssl-ibot-max 64 --ssl-ibot-sub 4 --ssl-ibot-zvar 1 --ssl-ibot-center 0
    --ssl-freeze-last 1000000 --ssl-dino 0 --ssl-koleo 0 --ssl-locals 0 --ssl-head-lr 1e-3 --ssl-head-warmup 500 --ssl-head-bottleneck 64
    --ssl-ibot-protos 1024 --attn-out-init 0.02 --teacher-temp0 0.02 --teacher-temp 0.02 --teacher-temp-warmup 3000"
}
P=${P:-512}
if [ ! -e $R/last.ckpt ] && [ "${BENCH:-1}" = 1 ]; then   # the window: 30 steps at P, else 384
  bash $D/mksources.sh $R/sources.json || exit 1
  ok=0; for p in $P 384; do
    b=$R/bench-$p; mkdir -p $b
    $U dino-train --gpu $GPU --sources $R/sources.json --out $b --steps 30 --save-every 1000000 --log-every 10 --eval-every 100000000 \
      --workers 12 --slots 8 --threads 4 $(cfg $p) > $b/log 2>&1 && grep -q "^step 30 " $b/log && { P=$p; ok=1; break; }
    echo "full.sh: P$p bench failed ($(tail -1 $b/log))"
  done
  [ $ok = 1 ] || { echo "full.sh: no window fits; see $R/bench-*/log"; exit 1; }
  echo "full.sh: P$P ($(grep -oE '[0-9]+ ms/step [0-9.]+ Mvox/s' $R/bench-$P/log | tail -1))"
fi
CFGX="$(cfg $P)" SLOTS=${SLOTS:-8} GPU=$GPU U=$U bash $D/launch.sh $R

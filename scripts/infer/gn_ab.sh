#!/usr/bin/env bash
# scripts/infer/gn_ab.sh CKPT TAG: running vs live GroupNorm statistics for a gn_ema checkpoint (audit 2026-11-14 infer #2; predict --gn-live 1).
# scratch2 trains with --gn-ema-warm 1e9, i.e. always on each window's own statistics, while predict normalises with the running ones. Two
# arms through the held-out evaluation (scripts/univ/eval.sh; holdout / unseen boxes only) plus a direct seam / difference probe on one box:
#   arm R (default, running statistics): TAG-gnrun          arm L (--gn-live 1): TAG-gnlive
# Outputs: $E/TAG-gnrun.matrix.txt, $E/TAG-gnlive.matrix.txt (eval.sh), $E/gnab-TAG/probe.txt (seam steps per arm, mean |d| between arms).
# Env: everything eval.sh takes (GPU, E, WB, SITE, U, LKF ...); TASKS (default "wind rv fib": eval.sh's ink rows still pass --window 544,
# which a 6-level latent net refuses; add ink once that is a multiple of 128), WIN / HALO (640 / 64: scratch2 needs multiples of 128),
# PROBE_BOX (level-2 PHerc1667 box z,y,x,nz,ny,nx; default a 1024 x 2048^2 block on the axis), AXIS (1667 axis json for the probe).
# Run after the MEGA test pass in a GPU window (one GPU, ~1-2 h for both arms at scratch2 size, est.).
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}; H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}
export TASKS=${TASKS:-wind rv fib} WIN=${WIN:-640} HALO=${HALO:-64} WZARGS=${WZARGS-}
E=${E:-/vesuvius/ufsm/wbench/univ}; O=$E/gnab-$T; mkdir -p $O
grep -q '"gn_ema":1' <(head -c 200000 $CK | head -1) || { echo "gn_ab: $CK has no running GroupNorm statistics (train --gn-ema): nothing to compare"; exit 2; }
for arm in run live; do
  if [ $arm = live ]; then export PX="--gn-live 1" PXH="--gn-live 1"; else export PX="${PX0:-}" PXH="${PXH0:-}"; fi
  [ -e $O/$arm.done ] || { E=$E bash $H/scripts/univ/eval.sh $CK $T-gn$arm > $O/eval_$arm.out 2>&1 && touch $O/$arm.done; } || echo "gn_ab: eval arm $arm failed ($O/eval_$arm.out)"
done
# the probe: one box, both arms, rv + ink + phase cos; seam steps at the window borders (ink-seam, period = the interior) and the arms' difference
R=${CTROOT:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}; K=${PROBE_KEY:-PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr}
B=${PROBE_BOX:-4096,1024,1536,1024,2048,2048}; AX=${AXIS:-}; AXA=(--no-axis 1); [ -n "$AX" ] && AXA=(--axis $AX)
for arm in run live; do X=(); [ $arm = live ] && X=(--gn-live 1)
  [ -e $O/p_$arm/rv/zarr.json ] || $U predict $CK $R $K $O/p_$arm --um 2.399 --level 2 --box $B --heads rv,ink,c0 --window $WIN --halo $HALO --shard 512 --q 0 --levels 1 \
       --gpu ${GPU:-0} --threads 8 --cache ${WB:-/vesuvius/ufsm/wbench}/cache "${AXA[@]}" "${X[@]}" > $O/p_$arm.log 2>&1 || echo "gn_ab: probe predict $arm failed ($O/p_$arm.log)"
done
st=$((WIN - 2 * HALO))
{ echo "gn_ab probe $T: box $B, window $WIN halo $HALO (interior $st)"
  for arm in run live; do for h in rv ink c0; do echo "seam $arm $h: $($U ink-seam $O/p_$arm/$h --um 9.596 --period $st --n 24 --min 8 --threads 8 2>&1 | tail -3 | tr '\n' ' ')"; done; done
  for h in rv ink c0; do echo "diff run vs live $h: $($U z3-diff $O/p_run/$h 9.596 $O/p_live/$h 9.596 --threads 8 2>&1 | tail -2 | tr '\n' ' ')"; done
  for arm in run live; do [ -e $E/$T-gn$arm.matrix.txt ] && { echo "== matrix $arm"; cat $E/$T-gn$arm.matrix.txt; }; done
} > $O/probe.txt
cat $O/probe.txt

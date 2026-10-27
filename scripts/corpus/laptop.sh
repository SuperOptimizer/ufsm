#!/usr/bin/env bash
# laptop.sh: the laptop side of the corpus unroll (see common.sh): the ift3-24000 ink of every scroll of list.tsv, in order, as soon as the
# desk has its box (desk <S>/box.txt). CT over HTTP from the public mirror (byte-identical to the desk copies; no --cache, so nothing of
# the CT lands on disk), plain one-pass predict (window 1024, halo 0: window-border seams; --occupancy 5 skips air), the ink zarr
# (~1-2 GB) rsynced to forlindesk, byte-checked -> forlindesk:<S>/state/ink.ready, then deleted here (laptop growth: one ink zarr).
# Each predict holds the laptop GPU through gpuq (prio bg) and runs in the memory slice (capped.sh, MEM 12G).
# Root C=~/ufsm-data/corpus (list.tsv, bin/ufsm = a master build, ckpt/ift3-24000.ckpt, <S>/state, logs). Stop: touch ~/ufsm-data/corpus/STOP.
# Start: cd ~/ufsm-data/corpus && setsid nohup bash <repo>/scripts/corpus/laptop.sh > laptop.out 2>&1 < /dev/null &
C=${C:-$HOME/ufsm-data/corpus}; H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
U=${LU:-$C/bin/ufsm}; INKCK=$C/ckpt/ift3-24000.ckpt; FD=${FDL:-forlindesk}   # (the laptop knows forlindesk by its ssh alias)
GPUQ_LOCK=$HOME/ufsm-data/gpu.lock; . $HOME/ufsm-data/wbench/bin/wb8/gpuq.sh
LIST=${ONLY:-$(scrolls)}

ink() {
  local box d=$FDC/$S a b
  box=$(ssh -n $DESK "cat $DC/$S/box.txt") || return 1; [ -n "$box" ] || return 1
  rm -rf $O/ink.zarr
  ( gpuq_acquire bg corpus-ink-$S
    MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U predict $INKCK $MIRROR $MK $O/ink.zarr --um $UM --level $LV --box $box --shard 1024 \
      --window 1024 --halo 0 --ema 1 --q 32 --threads 12 --gpu 0 --full-grid 1 --resume 1 --occupancy 5 --levels 4 > $O/logs/ink.log 2>&1
    rc=$?; gpuq_release; exit $rc ) || return 1
  ssh -n $FD "mkdir -p $d/state" && rsync -a $O/ink.zarr/ $FD:$d/ink.zarr/ || return 1
  a=$(bytes $O/ink.zarr); b=$(ssh -n $FD "du -sb --apparent-size $d/ink.zarr" | awk '{print $1}')
  [ "$a" = "$b" ] || { log "ink push: $a bytes here, ${b:-?} on forlindesk"; return 1; }
  ssh -n $FD "echo '$(date '+%F %T') $a ink bytes' > $d/state/ink.ready" || return 1
  rm -rf $O/ink.zarr; log "ink: $(awk -v b=$a 'BEGIN { printf "%.2f", b / 1e9 }') GB on forlindesk, deleted here"
}
while ! stopped; do
  left=0
  for S in $LIST; do
    scan $S || continue; failed && continue; done_ ink && continue; left=$((left + 1))
    ssh -n $DESK "test -s $DC/$S/box.txt" || continue
    run ink ink
  done
  [ $left = 0 ] && break; sleep 300
done
echo "$(date '+%m-%d %H:%M:%S') laptop.sh done" >> $C/corpus.log

#!/usr/bin/env bash
# laptop.sh: the laptop side of the corpus unroll (see common.sh): the ift3-24000 ink of every scroll of list.tsv, in order, as soon as the
# desk has its box (desk <S>/box.txt). CT over HTTP from the public mirror (byte-identical to the desk copies; no --cache, so nothing of
# the CT lands on disk), one predict pass (window 640, halo 64, 512^3 shards; 1152 pages on the 16 GB 5080: 5x slower; N measured the P4 border jump 30 (halo 0) -> 23 (frozen GN) -> 9.4 (halo 64 + frozen); --occupancy 5 skips air) with GroupNorm statistics frozen per scroll
# (a --gn-save calibration band first: per-window GN made tile-to-tile brightness seams on 1667). The ink zarr
# (~1-2 GB) rsynced to forlindesk, byte-checked -> forlindesk:<S>/state/ink.ready, then deleted here (laptop growth: one ink zarr).
# Each predict holds the laptop GPU through gpuq (prio bg) and runs in the memory slice (capped.sh, MEM 12G).
# Root C=~/ufsm-data/corpus (list.tsv, bin/ufsm = a master build, ckpt/ift3-24000.ckpt, <S>/state, logs). Stop: touch ~/ufsm-data/corpus/STOP.
# Start (from a copy, so repo edits never touch a running loop): cp <repo>/scripts/corpus/{common,laptop}.sh ~/ufsm-data/corpus/scripts/;
#   cd ~/ufsm-data/corpus && setsid nohup bash scripts/laptop.sh > laptop.out 2>&1 < /dev/null &
C=${C:-$HOME/ufsm-data/corpus}; H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
U=${LU:-$C/bin/ufsm}; INKCK=$C/ckpt/ift3-24000.ckpt; FD=${FDL:-forlindesk}   # (the laptop knows forlindesk by its ssh alias)
GPUQ_LOCK=$HOME/ufsm-data/gpu.lock; . $HOME/ufsm-data/wbench/bin/wb8/gpuq.sh
LIST=${ONLY:-$(scrolls)}

ink() {
  local box d=$FDC/$S a b
  box=$(ssh -n $DESK "cat $DC/$S/box.txt") || return 1; [ -n "$box" ] || return 1
  rm -rf $O/ink.zarr $O/gncal.zarr $O/gn.bin
  # GroupNorm calibration (predict --gn-save) over a 2048-voxel z band through the middle of the box (all of its y / x; air shards skipped),
  # then the ink pass with those frozen statistics (--gn-frozen): no per-window normalisation, so no tile-to-tile brightness jumps
  local cal=$(echo $box | awk -F, '{ zc = $1 + int($4 / 2); z0 = zc - 1024; if (z0 < $1) z0 = $1; nz = 2048; if (nz > $4) nz = $4; printf "%d,%d,%d,%d,%d,%d", z0, $2, $3, nz, $5, $6 }')
  ( gpuq_acquire bg corpus-ink-$S
    MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U predict $INKCK $MIRROR $MK $O/gncal.zarr --um $UM --level $LV --box $cal --shard 512 \
      --window 640 --halo 64 --ema 1 --q 32 --threads 12 --gpu 0 --full-grid 1 --occupancy 5 --levels 1 --no-out 1 --gn-save $O/gn.bin > $O/logs/ink-gn.log 2>&1 &&
    MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U predict $INKCK $MIRROR $MK $O/ink.zarr --um $UM --level $LV --box $box --shard 512 \
      --window 640 --halo 64 --ema 1 --q 32 --threads 12 --gpu 0 --full-grid 1 --resume 1 --occupancy 5 --levels 4 --gn-frozen $O/gn.bin > $O/logs/ink.log 2>&1
    rc=$?; gpuq_release; exit $rc ) || return 1
  rm -rf $O/gncal.zarr
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

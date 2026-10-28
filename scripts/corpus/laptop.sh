#!/usr/bin/env bash
# laptop.sh: the laptop side of the corpus unroll (see common.sh): the ift3-24000 ink of every scroll of list.tsv, in order, as soon as the
# desk has its box (desk <S>/box.txt). CT over HTTP from the public mirror (byte-identical to the desk copies; no --cache, so nothing of
# the CT lands on disk). GroupNorm statistics frozen per scroll (a --gn-save calibration band first: per-window GN made tile-to-tile
# brightness seams on 1667), then two predict passes (window 640, halo 64, 512^3 shards; --occupancy 5 skips air): A on the shard grid,
# B shifted 256 voxels in z / y / x, blended by ufsm ink-blend (sin^2 tile weights). One pass left window-border jumps of ~24-27 (ufsm
# ink-seam; halo 128 or window 1152 still 15-25); the blend brings them to the ~5.5 of planes away from any border (0125 A/B 10-27).
# The blended ink zarr (~1-2 GB) is rsynced to the desk (renders run there), byte-checked -> desk <S>/state/ink.ready, then deleted here
# (laptop growth: the two passes + the blend, ~3x one ink zarr, at most).
# Each predict holds the laptop GPU through gpuq (prio bg) and runs in the memory slice (capped.sh, MEM 12G).
# Root C=~/ufsm-data/corpus (list.tsv, bin/ufsm = a master build, ckpt/ift3-24000.ckpt, <S>/state, logs). Stop: touch ~/ufsm-data/corpus/STOP.
# Start (from a copy, so repo edits never touch a running loop): cp <repo>/scripts/corpus/{common,laptop}.sh ~/ufsm-data/corpus/scripts/;
#   cd ~/ufsm-data/corpus && setsid nohup bash scripts/laptop.sh > laptop.out 2>&1 < /dev/null &
C=${C:-$HOME/ufsm-data/corpus}; H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
U=${LU:-$C/bin/ufsm}; INKCK=$C/ckpt/ift3-24000.ckpt
GPUQ_LOCK=$HOME/ufsm-data/gpu.lock; . $HOME/ufsm-data/wbench/bin/wb8/gpuq.sh
LIST=${ONLY:-$(scrolls)}

ink() {
  local box d=$DC/$S a b bb bo
  box=$(ssh -n $DESK "cat $DC/$S/box.txt") || return 1; [ -n "$box" ] || return 1
  rm -rf $O/gncal.zarr   # (a retry keeps gn.bin and the written shards of both passes: --resume; all go only after a verified push)
  local cal=$(echo $box | awk -F, '{ zc = $1 + int($4 / 2); z0 = zc - 1024; if (z0 < $1) z0 = $1; nz = 2048; if (nz > $4) nz = $4; printf "%d,%d,%d,%d,%d,%d", z0, $2, $3, nz, $5, $6 }')
  bb=$(echo $box | awk -F, '{ printf "%d,%d,%d,%d,%d,%d", $1 + 256, $2 + 256, $3 + 256, $4 - 256, $5 - 256, $6 - 256 }'); bo=$(echo $bb | cut -d, -f1-3)
  if [ ! -e $O/blend.done ]; then
    ( gpuq_acquire bg corpus-ink-$S
      { [ -s $O/gn.bin ] || { MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U predict $INKCK $MIRROR $MK $O/gncal.zarr --um $UM --level $LV --box $cal --shard 512 \
        --window 640 --halo 64 --ema 1 --q 32 --threads 12 --gpu 0 --full-grid 1 --occupancy 5 --levels 1 --no-out 1 --gn-save $O/gn.tmp > $O/logs/ink-gn.log 2>&1 && mv $O/gn.tmp $O/gn.bin; }; } &&
      MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U predict $INKCK $MIRROR $MK $O/ink.zarr --um $UM --level $LV --box $box --shard 512 \
        --window 640 --halo 64 --ema 1 --q 32 --threads 12 --gpu 0 --full-grid 1 --resume 1 --occupancy 5 --levels 4 --gn-frozen $O/gn.bin >> $O/logs/ink.log 2>&1 &&
      MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U predict $INKCK $MIRROR $MK $O/inkb.zarr --um $UM --level $LV --box $bb --shard 512 \
        --window 640 --halo 64 --ema 1 --q 32 --threads 12 --gpu 0 --resume 1 --occupancy 5 --levels 1 --gn-frozen $O/gn.bin >> $O/logs/inkb.log 2>&1
      rc=$?; gpuq_release; exit $rc ) || return 1
    rm -rf $O/gncal.zarr $O/inkbl.zarr
    MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 $U ink-blend $O/inkbl.zarr --um $UM --tile 512,512,512 --grids 0,0,0:256,256,256 --origins 0,0,0:$bo \
      --levels 4 --shard 512 --q 32 --threads 12 $O/ink.zarr $O/inkb.zarr > $O/logs/blend.log 2>&1 || return 1
    touch $O/blend.done; rm -rf $O/ink.zarr $O/inkb.zarr
  fi
  ssh -n $DESK "mkdir -p $d/state; rm -f $d/state/ink.ready" && rsync -a --delete $O/inkbl.zarr/ $DESK:$d/ink.zarr/ || return 1
  a=$(bytes $O/inkbl.zarr); b=$(ssh -n $DESK "du -sb --apparent-size $d/ink.zarr" | awk '{print $1}')
  [ "$a" = "$b" ] || { log "ink push: $a bytes here, ${b:-?} on the desk"; return 1; }
  ssh -n $DESK "echo '$(date '+%F %T') $a ink bytes (2-pass blend)' > $d/state/ink.ready" || return 1
  rm -rf $O/inkbl.zarr $O/gn.bin $O/blend.done; log "ink: $(awk -v b=$a 'BEGIN { printf "%.2f", b / 1e9 }') GB (2-pass blend) on the desk, deleted here"
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

#!/usr/bin/env bash
# laptop.sh: the laptop ink lane of the corpus unroll (common.sh; the 2-pass ift3 ink and the claims: inklib.sh). Scrolls as soon as the
# desk has their box (desk <S>/box.txt), claimed on forlindesk against the forlindesk ink lanes (fdink.sh); CT over HTTP from the public
# mirror (no CT copy, no chunk cache). The blended ink zarr (~1-2 GB) is rsynced to the desk (renders run there), byte-checked -> desk
# <S>/state/ink.ready + ink.2pass (and a re-ink's render / clean redone), then deleted here (laptop growth: the two passes + the blend).
# Each predict holds the laptop GPU through gpuq (prio bg) and runs in the memory slice (capped.sh, MEM 12G).
# Root C=~/ufsm-data/corpus (list.tsv, bin/ufsm = a master build, ckpt/ift3-24000.ckpt, <S>/state, logs). Stop: touch ~/ufsm-data/corpus/STOP.
# Start (from a copy, so repo edits never touch a running loop): cp <repo>/scripts/corpus/{common,inklib,laptop}.sh ~/ufsm-data/corpus/scripts/;
#   cd ~/ufsm-data/corpus && setsid nohup bash scripts/laptop.sh > laptop.out 2>&1 < /dev/null &
FD=${FD:-forrest@forlindesk}   # (the laptop knows forlindesk by name, not by the desk's LAN address)
C=${C:-$HOME/ufsm-data/corpus}; H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh; . $H/inklib.sh
U=${LU:-$C/bin/ufsm}; INKCK=$C/ckpt/ift3-24000.ckpt; ME=laptop; GPU=0; CTLOCAL=0
fdsh() { ssh -n $FD "$1"; }
gpu_run() { MEM=${MEM:-12G} $HOME/ufsm-data/wbench/bin/wb8/gpuq.sh run bg corpus-ink-$S $HOME/ufsm-data/capped.sh nice -n 5 "$@"; }
cpu_run() { MEM=${MEM:-12G} $HOME/ufsm-data/capped.sh nice -n 5 "$@"; }
deskbox() { ssh -n $DESK "cat $DC/$S/box.txt"; }
push() {
  local d=$DC/$S a b
  ssh -n $DESK "mkdir -p $d/state; rm -f $d/state/ink.ready $d/state/ink.2pass" && rsync -a --delete $O/inkbl.zarr/ $DESK:$d/ink.zarr/ || return 1
  a=$(bytes $O/inkbl.zarr); b=$(ssh -n $DESK "du -sb --apparent-size $d/ink.zarr" | awk '{print $1}')
  [ "$a" = "$b" ] || { log "ink push: $a bytes here, ${b:-?} on the desk"; return 1; }
  ssh -n $DESK "echo '$(date '+%F %T') $a ink bytes ($ME)' > $d/state/ink.ready; cp $d/state/ink.ready $d/state/ink.2pass; rm -f $d/state/render.done $d/state/clean.done" || return 1
  rm -rf $O/inkbl.zarr $O/gn.bin $O/blend.done; log "ink: $(awk -v b=$a 'BEGIN { printf "%.2f", b / 1e9 }') GB on the desk, deleted here"
}
ink_loop deskbox push
echo "$(date '+%m-%d %H:%M:%S') laptop.sh done" >> $C/corpus.log

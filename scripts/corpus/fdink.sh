#!/usr/bin/env bash
# fdink.sh GPU: a forlindesk ink lane (one per 3090; common.sh, inklib.sh): the 2-pass ift3 ink of scrolls whose box is here (the desk
# pushed it with the fields), claimed against the laptop and the other card, CT over HTTP from the mirror. It fills the gaps of fd.sh's
# slab fits: a predict starts only after no `ufsm fit --gpu GPU` has run on the card for IDLE (3) minutes, and is killed when one starts
# (predict --resume continues it later). Output: <S>/inkbl.zarr + state/ink.fdready here; the desk's inkpull lane fetches it (forlindesk
# cannot ssh to the desk), then deletes it here. Needs the sm_86 build (make ARCH=86) and ckpt/ift3-24000.ckpt under C.
# Start (from a copy): cp src/scripts/corpus/{common,inklib,fdink}.sh runink/; for g in 0 1; do
#   STOPF=/vesuvius/ufsm/corpus/STOPINK setsid nohup bash runink/fdink.sh $g > fdink$g.out 2>&1 < /dev/null & done
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh; . $H/inklib.sh
GPU=${1:?GPU}; ME=fd-g$GPU; INKCK=${INKCK_FD:-$C/ckpt/ift3-24000.ckpt}; CTLOCAL=0; STOPF=${STOPF:-$C/STOPINK}
fdsh() { bash -c "$1"; }
fitbusy() { pgrep -f "ufsm fi[t] .*--gpu $GPU( |$)" > /dev/null; }
gpu_run() {   # wait for IDLE fit-free minutes on the card, run; a slab fit appearing kills the command and the wait starts again
  local p n
  while :; do
    n=0; while [ $n -lt ${IDLE:-3} ]; do stopped && return 1; if fitbusy; then n=0; else n=$((n + 1)); fi; sleep 60; done
    nice -n 5 "$@" & p=$!
    while kill -0 $p 2> /dev/null; do
      sleep 20
      if fitbusy || stopped; then kill $p 2> /dev/null; wait $p; log "ink: yields GPU $GPU (slab fits / stop)"; stopped && return 1; continue 2; fi
    done
    wait $p; return $?
  done
}
cpu_run() { nice -n 10 "$@"; }
fdbox() { cat $O/box.txt 2> /dev/null; }
push() { echo "$(date '+%F %T') $(bytes $O/inkbl.zarr) ink bytes ($ME)" > $O/state/ink.fdready; log "ink: ready here for the desk ($ME)"; }
ink_loop fdbox push

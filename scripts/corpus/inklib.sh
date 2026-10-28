# scripts/corpus/inklib.sh: the 2-pass ift3-24000 ink of one scroll, shared by the ink lanes (laptop.sh; fdink.sh: one per forlindesk
# 3090; sourced after common.sh). The caller sets ME (the claim owner: laptop, fd-g0, ..), GPU, INKCK, CTLOCAL (1: the CT from the local
# list.tsv root, else the mirror over HTTP) and defines fdsh CMD (a shell on forlindesk), gpu_run CMD.. (one GPU command: its queue /
# preemption) and cpu_run CMD.. (the CPU blend).
# Claims: atomic mkdir forlindesk:<S>/state/ink.lock (+ owner), reachable from every ink host (forlindesk cannot ssh to the desk; the
# desk pulls forlindesk's inks). Order: scrolls without ink first, then those marked forlindesk:<S>/state/ink.redo (single-pass inks).
# Ink: GroupNorm statistics frozen per scroll (predict --gn-save over a 2048-voxel z band through the middle of the box), then pass A
# (window 640 halo 64, 512^3 shards, --full-grid) and pass B (the box + 256 voxels in z / y / x, box-local), blended by ufsm ink-blend
# (sin^2 tile weights): window-border jumps 26.7 -> 5.8 on 0125 (= the 5.5 of planes away from any border; ufsm ink-seam). Every
# predict --resumes, so a killed pass continues where it stopped.
ink_lock() {   # this host owns S's ink (taken now, or earlier by this host)
  local o; o=$(fdsh "d=$FDC/$S/state; mkdir -p \$d; mkdir \$d/ink.lock 2> /dev/null && echo $ME > \$d/ink.lock/owner; cat \$d/ink.lock/owner 2> /dev/null") || return 1
  [ "$o" = "$ME" ]
}
ink_cands() {   # the scrolls this host may ink, in order: free or own claim, not finished (ink.done here), fresh before ink.redo
  local s st
  fdsh "cd $FDC; for s in $(scrolls | tr '\n' ' '); do o=\$(cat \$s/state/ink.lock/owner 2> /dev/null); r=; [ -e \$s/state/ink.redo ] && r=redo; echo \$s \${o:-free} \${r:-fresh}; done" |
    awk -v me=$ME '($2 == "free" || $2 == me) { if ($3 == "fresh") print $1; else redo = redo " " $1 } END { n = split(redo, a, " "); for (i = 1; i <= n; i++) print a[i] }'
}
ink_compute() {   # ink_compute BOX -> $O/inkbl.zarr (+ blend.done); resumable at every step
  local box=$1 cal bb bo ctr=$MIRROR ctk=$MK; [ "${CTLOCAL:-0}" = 1 ] && { ctr=$R; ctk=$K; }   # (the CT: the mirror over HTTP, or the local copy)
  [ -e $O/blend.done ] && return 0
  cal=$(echo $box | awk -F, '{ zc = $1 + int($4 / 2); z0 = zc - 1024; if (z0 < $1) z0 = $1; nz = 2048; if (nz > $4) nz = $4; printf "%d,%d,%d,%d,%d,%d", z0, $2, $3, nz, $5, $6 }')
  bb=$(echo $box | awk -F, '{ printf "%d,%d,%d,%d,%d,%d", $1 + 256, $2 + 256, $3 + 256, $4 - 256, $5 - 256, $6 - 256 }'); bo=$(echo $bb | cut -d, -f1-3)
  rm -rf $O/gncal.zarr
  if [ ! -s $O/gn.bin ]; then
    gpu_run $U predict $INKCK $ctr $ctk $O/gncal.zarr --um $UM --level $LV --box $cal --shard 512 --window 640 --halo 64 --ema 1 --q 32 --threads 12 \
      --gpu $GPU --full-grid 1 --occupancy 5 --levels 1 --no-out 1 --gn-save $O/gn.tmp > $O/logs/ink-gn.log 2>&1 && mv $O/gn.tmp $O/gn.bin || return 1
  fi
  gpu_run $U predict $INKCK $ctr $ctk $O/ink.zarr --um $UM --level $LV --box $box --shard 512 --window 640 --halo 64 --ema 1 --q 32 --threads 12 \
    --gpu $GPU --full-grid 1 --resume 1 --occupancy 5 --levels 4 --gn-frozen $O/gn.bin >> $O/logs/ink.log 2>&1 || return 1
  gpu_run $U predict $INKCK $ctr $ctk $O/inkb.zarr --um $UM --level $LV --box $bb --shard 512 --window 640 --halo 64 --ema 1 --q 32 --threads 12 \
    --gpu $GPU --resume 1 --occupancy 5 --levels 1 --gn-frozen $O/gn.bin >> $O/logs/inkb.log 2>&1 || return 1
  rm -rf $O/gncal.zarr $O/inkbl.zarr
  cpu_run $U ink-blend $O/inkbl.zarr --um $UM --tile 512,512,512 --grids 0,0,0:256,256,256 --origins 0,0,0:$bo --levels 4 --shard 512 --q 32 --threads 12 \
    $O/ink.zarr $O/inkb.zarr > $O/logs/blend.log 2>&1 || return 1
  touch $O/blend.done; rm -rf $O/ink.zarr $O/inkb.zarr
}
ink_loop() {   # ink_loop BOXCMD PUSHFN: claims and inks scrolls until none is left; BOXCMD prints S's box, PUSHFN delivers $O/inkbl.zarr
  local boxcmd=$1 push=$2 S0 box k n
  while ! stopped; do
    n=0
    for S0 in $(ink_cands); do
      S=$S0; scan $S || continue; [ -e $O/state/ink.done ] || [ -e $O/state/ink.failed ] && continue
      box=$($boxcmd) && [ -n "$box" ] || continue
      ink_lock || continue
      n=$((n + 1))
      for k in $(seq 0 ${RETRY:-2}); do
        log "ink start (try $k, $ME)"; local t0=$(date +%s)
        if ink_compute $box && $push; then echo "$(date '+%F %T') $(( $(date +%s) - t0 )) s" > $O/state/ink.done; log "ink done ($ME) $(( $(date +%s) - t0 )) s"; break; fi
        log "ink FAILED (try $k, $ME, logs/ink*)"; stopped && return 0; sleep 60
        [ $k = ${RETRY:-2} ] && { touch $O/state/ink.failed; log "ink: giving up ($ME)"; }
      done
      break   # (re-read the candidates: other hosts claim meanwhile)
    done
    [ $n = 0 ] && sleep 300
  done
}

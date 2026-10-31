# scripts/corpus/inklib.sh: the seam-free single-pass ift3-24000 ink of one scroll, shared by the ink lanes (laptop.sh; fdink.sh: one per forlindesk
# 3090; sourced after common.sh). The caller sets ME (the claim owner: laptop, fd-g0, ..), GPU, INKCK, CTLOCAL (1: the CT from the local
# list.tsv root, else the mirror over HTTP) and defines fdsh CMD (a shell on forlindesk), gpu_run CMD.. (one GPU command: its queue /
# preemption) and cpu_run CMD.. (the CPU blend).
# Claims: atomic mkdir forlindesk:<S>/state/ink.lock (+ owner), reachable from every ink host (forlindesk cannot ssh to the desk; the
# desk pulls forlindesk's inks). Order: scrolls without ink first, then those marked forlindesk:<S>/state/ink.redo (single-pass inks).
# Ink: GroupNorm statistics frozen per scroll (predict --gn-save over a 2048-voxel z band through the middle of the box), then ONE pass
# with feathered windows (predict --feather 32: window 576 halo 32, 512^3 shards, --full-grid; each window border blended over 32 voxels
# from the earlier window's halo prediction) straight into inkbl.zarr. Paris 4 crops (10-30, ufsm ink-seam --block 16): border jump
# 20.2 (halo 0) / 6.3 (halo 64) -> 0.0, the same as the old 2-pass A + B + ink-blend (0.0) at ~1/3 of its GPU time and no blend.
# INKW / INKH / INKF override window / halo / feather (e.g. 1088 / 32 / 32 on 24 GB GPUs: interior 1024, 1.2x a halo-0 pass).
# The predict --resumes, so a killed pass continues where it stopped (the done windows' spills are recomputed).
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
  rm -rf $O/gncal.zarr
  if [ ! -s $O/gn.bin ]; then
    gpu_run $U predict $INKCK $ctr $ctk $O/gncal.zarr --um $UM --level $LV --box $cal --shard 512 --window 640 --halo 64 --ema 1 --q 32 --threads 12 \
      --gpu $GPU --full-grid 1 --occupancy 5 --levels 1 --no-out 1 --gn-save $O/gn.tmp > $O/logs/ink-gn.log 2>&1 && mv $O/gn.tmp $O/gn.bin || return 1
  fi
  gpu_run $U predict $INKCK $ctr $ctk $O/inkbl.zarr --um $UM --level $LV --box $box --shard 512 --window ${INKW:-auto} --halo ${INKH:-32} --feather ${INKF:-32} \
    --ema 1 --q 32 --threads 12 --gpu $GPU --full-grid 1 --resume 1 --occupancy 5 --levels 4 --gn-frozen $O/gn.bin >> $O/logs/ink.log 2>&1 || return 1
  rm -rf $O/gncal.zarr
  touch $O/blend.done
}
ink_loop() {   # ink_loop BOXCMD PUSHFN: claims and inks scrolls until none is left; BOXCMD prints S's box, PUSHFN delivers $O/inkbl.zarr
  local boxcmd=$1 push=$2 S0 box k n left
  while ! stopped; do
    n=0; left=0
    for S0 in $(ink_cands); do
      S=$S0; scan $S || continue; [ -e $O/state/ink.done ] || [ -e $O/state/ink.failed ] && continue
      left=$((left + 1))
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
    [ $left = 0 ] && return 0   # (every scroll this host may ink is inked or failed: the lane ends)
    [ $n = 0 ] && sleep 300
  done
}

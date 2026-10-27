#!/usr/bin/env bash
# desk.sh: the desk side of the corpus unroll (see common.sh): for every scroll of list.tsv in order: prep -> wind96 fields (GPU 0) ||
# ift3-24000 ink (GPU 1) -> push to forlindesk in the background (one push at a time) while the next scroll predicts.
# Start: cd /vesuvius/ufsm/corpus && setsid nohup bash src/scripts/corpus/desk.sh > desk.out 2>&1 < /dev/null &
# Env: ONLY ("S1 S2": these scrolls), SPACEGB (120 free desk GB before a predict), FDGB (250: free forlindesk GB before a push),
#   INK (1), WAITGPU (1: wait until no other process holds a desk GPU before each scroll's predicts).
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh

prep() {   # box of the scanprep mask (object cells, padded one cell, 64-aligned, level voxels) and the axis
  local id=$(basename $K | cut -c1-14) m sh GZ GY GX cell
  m=/vesuvius/ufsm/u0/prep/$S-$id.mask.raw; sh=$(jq -r '.shape | join(" ")' $R/$K/$LV/zarr.json)
  read -r GZ GY GX < <(jq -r '.grid | join(" ")' $m.json); cell=$(jq -r .cell $m.json)
  od -An -v -tu1 -w$((GY * GX)) $m | awk -v X=$GX -v c=$cell -v sh="$sh" '
    { for (i = 1; i <= NF; i++) if ($i > 0) { y = int((i - 1) / X); x = (i - 1) % X; z = NR - 1
        if (!n++) { z0 = z1 = z; y0 = y1 = y; x0 = x1 = x }
        if (z < z0) z0 = z; if (z > z1) z1 = z; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x } }
    END { split(sh, s, " "); a[1] = z0; a[2] = y0; a[3] = x0; b[1] = z1; b[2] = y1; b[3] = x1
      for (d = 1; d <= 3; d++) { lo = (a[d] - 1) * c; if (lo < 0) lo = 0; lo = int(lo / 64) * 64; hi = (b[d] + 2) * c; if (hi > s[d]) hi = s[d]; o[d] = lo; e[d] = hi - lo }
      printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], e[1], e[2], e[3] }' > $O/box.txt
  [ -s $O/box.txt ] || return 1
  # axis: an umbilicus in the scan's own frame (list.tsv) or ufsm axis (per-z centroid of the masked CT)
  if [ "$AXS" = auto ]; then N $U axis $R $K $O/axis.json > $O/logs/axis.log 2>&1 || return 1; else cp $AXS $O/axis.json; fi
  log "box $(cat $O/box.txt) (level $LV, $LUM um), axis $AXS"
}
fields() {
  space || return 1; rm -rf $F $F.tmp
  N $U predict $CK $R $K $F.tmp --um $UM --level $LV --box $(cat $O/box.txt) --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads 16 --gpu 0 --axis $O/axis.json --levels 1 --fit-zarr $F --fit-h 2 --fit-q 8 --fit-q-phase 0 \
    --no-out 1 > $O/logs/fields.log 2>&1; local rc=$?; rm -rf $F.tmp; [ $rc = 0 ] && [ -e $F/fields.json ]
}
ink() {   # plain (one pass, halo 0) ift3 predict of the box: has window-border seams (N: ink-blend of 8 offset passes fixes them, 8x cost)
  [ "${INK:-1}" = 1 ] || return 0
  N $U predict $INKCK $R $K $O/ink.zarr --um $UM --level $LV --box $(cat $O/box.txt) --shard 1024 --window 1024 --halo 0 --ema 1 --q 32 --threads 16 \
    --gpu 1 --full-grid 1 --resume 1 --occupancy 5 --levels 4 >> $O/logs/ink.log 2>&1
}
bytes() { du -sb --apparent-size "$@" | awk '{s += $1} END {print s + 0}'; }
push() {   # fields, ink, the CT level, box + axis -> forlindesk C/<S>; byte-checked; then forlindesk:<S>/state/ready and the desk fields deleted
  local f=$(ssh -n $FD "df -BG --output=avail $C | tail -n 1" | tr -dc 0-9) d=$C/$S cn=$(basename $K) a b
  [ "${f:-0}" -ge ${FDGB:-250} ] || { log "push: forlindesk has ${f:-?} GB free (FDGB ${FDGB:-250})"; return 1; }
  ssh -n $FD "mkdir -p $d/state $d/logs $d/ct/$cn" || return 1
  rsync -a $O/box.txt $O/axis.json $FD:$d/ && rsync -a $F/ $FD:$d/fields/ && { [ "${INK:-1}" != 1 ] || rsync -a $O/ink.zarr/ $FD:$d/ink.zarr/; } &&
    rsync -a $R/$K/zarr.json $FD:$d/ct/$cn/ && rsync -a $R/$K/$LV/ $FD:$d/ct/$cn/$LV/ || return 1
  for p in "$F fields" "$O/ink.zarr ink.zarr" "$R/$K/$LV ct/$cn/$LV"; do set -- $p; [ -e $1 ] || continue
    a=$(bytes $1); b=$(ssh -n $FD "du -sb --apparent-size $d/$2" | awk '{print $1}')
    [ "$a" = "$b" ] || { log "push: $2 $a bytes here, ${b:-?} on forlindesk"; return 1; }; done
  ssh -n $FD "echo '$(date '+%F %T') $(bytes $F) field bytes' > $d/state/ready" || return 1
  rm -rf $F; log "push: $(awk -v b=$(ssh -n $FD "du -sb $d" | awk '{print $1}') 'BEGIN { printf "%.1f", b / 1e9 }') GB on forlindesk, desk fields deleted"
}

LIST=${ONLY:-$(tail -n +2 $C/list.tsv | cut -f1)}; PP=
for S in $LIST; do
  stopped && break; scan $S || continue; [ -e $O/state/failed ] && { log "failed before, skipped"; continue; }
  done_ push && continue
  run prep prep || continue
  if ! done_ fields || ! done_ ink; then
    while [ "${WAITGPU:-1}" = 1 ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done   # (our own predicts have ended here)
    ( run ink ink ) & IP=$!
    run fields fields; rf=$?; wait $IP; ri=$?
    [ $rf = 0 ] && [ $ri = 0 ] || continue
  fi
  [ -n "$PP" ] && wait $PP   # one push at a time
  ( run push push ) & PP=$!
done
[ -n "$PP" ] && wait $PP
ssh -n $FD "touch $C/desk.done"
echo "$(date '+%m-%d %H:%M:%S') desk.sh done" >> $C/corpus.log

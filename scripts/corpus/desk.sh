#!/usr/bin/env bash
# desk.sh: the desk side of the corpus unroll (see common.sh). Lanes (one process; restart-safe: done markers + flock claims):
#   prep     every scroll up front, PREPPAR (3) at once: box of the scanprep mask (object cells padded one cell, origin on the 512-voxel grid: predict --fit-zarr slab rows are 256 cells = 512 voxels),
#            axis (list.tsv umbilicus or ufsm axis = per-z centroid of the masked CT)
#   gpu 0/1  wind96 fit fields of one scroll per GPU (run3-cf predict settings) + lossless repack to 512^3 shards; waits for SPACEGB
#   ref      count-field ref on the CPU (REFPAR (2) lanes, TH (12) threads each, nice 10 under the predicts; MemAvailable >= MEMGB 40 first)
#   push     fields + ref + box + axis -> forlindesk (FDGB 250 free there), byte-checked -> forlindesk:<S>/state/fields.ready; desk fields deleted
# Start: cd /vesuvius/ufsm/corpus && setsid nohup bash src/scripts/corpus/desk.sh > desk.out 2>&1 < /dev/null &
# Env: ONLY ("S1 S2"), SPACEGB (150: free desk GB before a predict), GPUS ("0 1"), WAITGPU (1: a lane waits until its GPU holds no other process).
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
LIST=${ONLY:-$(scrolls)}

prep() {
  local id=$(basename $K | cut -c1-14) m sh GZ GY GX cell
  m=/vesuvius/ufsm/u0/prep/$S-$id.mask.raw; sh=$(jq -r '.shape | join(" ")' $R/$K/$LV/zarr.json)
  read -r GZ GY GX < <(jq -r '.grid | join(" ")' $m.json); cell=$(jq -r .cell $m.json)
  od -An -v -tu1 -w$((GY * GX)) $m | awk -v X=$GX -v c=$cell -v sh="$sh" '
    { for (i = 1; i <= NF; i++) if ($i > 0) { y = int((i - 1) / X); x = (i - 1) % X; z = NR - 1
        if (!n++) { z0 = z1 = z; y0 = y1 = y; x0 = x1 = x }
        if (z < z0) z0 = z; if (z > z1) z1 = z; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x } }
    END { split(sh, s, " "); a[1] = z0; a[2] = y0; a[3] = x0; b[1] = z1; b[2] = y1; b[3] = x1
      for (d = 1; d <= 3; d++) { lo = (a[d] - 1) * c; if (lo < 0) lo = 0; lo = int(lo / 512) * 512; hi = (b[d] + 2) * c; if (hi > s[d]) hi = s[d]; o[d] = lo; e[d] = hi - lo }
      printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], e[1], e[2], e[3] }' > $O/box.txt
  [ -s $O/box.txt ] || return 1
  if [ "$AXS" = auto ]; then MEMGB=60 memok; N $U axis $R $K $O/axis.json > $O/logs/axis.log 2>&1 || return 1; else cp $AXS $O/axis.json; fi
  log "box $(cat $O/box.txt) (level $LV, $LUM um), axis $AXS"
}
fields() {   # (GPU $G)
  rm -rf $F $F.tmp
  N $U predict $CK $R $K $F.tmp --um $UM --level $LV --box $(cat $O/box.txt) --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads 12 --gpu $G --axis $O/axis.json --levels 1 --fit-zarr $F --fit-h 2 --fit-q 8 --fit-q-phase 0 \
    --no-out 1 > $O/logs/fields.log 2>&1 || return 1
  rm -rf $F.tmp; [ -e $F/fields.json ] || return 1
  local a b; for a in $F/*/zarr.json; do   # predict --fit-zarr writes 128^3 shards: lossless repack to 512^3 (64x fewer files)
    jq -e '.chunk_grid.configuration.chunk_shape[0] == 512' $a > /dev/null && continue; b=$(dirname $a)
    rm -rf $b.512 && N $U repack $b $b.512 512 --threads 12 > /dev/null && rm -rf $b && mv $b.512 $b || return 1
  done
  log "fields $(du -sh $F | cut -f1)"
}
ref() {
  rm -rf $REF; memok
  /usr/bin/time -v nice -n 10 $U count-field $F $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $REF --model-only 1 --model-stride 2 \
    --threads ${TH:-12} --mem-gb ${CFMEM:-24} > $O/logs/ref.log 2>&1 && [ -e $REF/model.json ]
}
push() {
  local f=$(ssh -n $FD "df -BG --output=avail $FDC | tail -n 1" | tr -dc 0-9) d=$FDC/$S a b p
  [ "${f:-0}" -ge ${FDGB:-250} ] || { log "push: forlindesk has ${f:-?} GB free (FDGB ${FDGB:-250})"; return 1; }
  ssh -n $FD "mkdir -p $d/state $d/logs" && rsync -a $O/box.txt $O/axis.json $FD:$d/ && rsync -a $F/ $FD:$d/fields/ && rsync -a $REF/ $FD:$d/ref/ || return 1
  for p in fields ref; do a=$(bytes $O/$p); b=$(ssh -n $FD "du -sb --apparent-size $d/$p" | awk '{print $1}')
    [ "$a" = "$b" ] || { log "push: $p $a bytes here, ${b:-?} on forlindesk"; return 1; }; done
  ssh -n $FD "echo '$(date '+%F %T') $(bytes $F) field bytes' > $d/state/fields.ready" || return 1
  rm -rf $F; log "push: fields + ref on forlindesk ($(awk -v b=$a 'BEGIN { printf "%.1f", b / 1e9 }') GB ref), desk fields deleted"
}
# a lane: LANE STAGE NEED FN [G]: every scroll of LIST whose NEED stage is done (or NEED = -) and STAGE is not, claimed; until none is left
lane() {
  local st=$1 need=$2 fn=$3 left; G=${4:-}
  while ! stopped; do
    left=0
    for S in $LIST; do
      scan $S || continue; failed && continue; done_ $st && continue; left=$((left + 1))
      [ "$need" = - ] || done_ $need || continue
      claim $st || continue
      if [ $st = fields ]; then
        while [ "${WAITGPU:-1}" = 1 ] && nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
        while [ $(freegb) -lt ${SPACEGB:-150} ]; do log "fields waits: $(freegb) GB free (SPACEGB ${SPACEGB:-150})"; sleep 600; done
      fi
      run $st $fn; unclaim
    done
    [ $left = 0 ] && break; sleep 60
  done
}

for S in $LIST; do scan $S; (failed || done_ prep) && continue; ( run prep prep ) &
  while [ $(jobs -r | wc -l) -ge ${PREPPAR:-3} ]; do sleep 5; done; done; wait
for g in ${GPUS:-0 1}; do lane fields prep fields $g & done
for k in $(seq 1 ${REFPAR:-2}); do lane ref fields ref & sleep 5; done
lane push ref push & wait
ssh -n $FD "touch $FDC/desk.done"
echo "$(date '+%m-%d %H:%M:%S') desk.sh done" >> $C/corpus.log

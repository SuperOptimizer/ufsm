#!/usr/bin/env bash
# desk.sh: the desk side of the corpus unroll (see common.sh). Lanes (one process; restart-safe: done markers + flock claims):
#   prep     every scroll up front, PREPPAR (3) at once: box of the scanprep mask (object cells padded one cell, origin on the 512-voxel
#            grid: predict --fit-zarr slab rows are 256 cells = 512 voxels), axis (list.tsv umbilicus or ufsm axis = per-z CT centroid)
#   gpu 0/1  wind96 fit fields of one scroll per GPU (run3-cf predict settings) + lossless repack to 512^3 shards; waits for SPACEGB
#   push     fields + box + axis -> forlindesk as soon as the fields are done (FDGB 250 free there), byte-checked ->
#            forlindesk:<S>/state/fields.ready (fields kept here); forlindesk computes the count-field ref itself before its slabs
#   ref      fallback only (one lane): when forlindesk holds more than one pushed scroll without wraps, the desk takes a queued scroll's
#            ref (atomic lock forlindesk:<S>/state/ref.lock, owner desk), computes it here (TH 12 threads, nice 10), pushes ref/ and writes
#            forlindesk:<S>/state/ref.ready; --model-stride MSTRIDE (4): Paris 4 at 4 vs 2 (forlindesk, 10 threads each): 1:36 vs 1:54 h,
#            ref-alone ids .1598 vs .1605, pairs .691/.295 vs .683/.296
#   pull     once forlindesk:<S>/state/wraps.ready: its joined model + wraps -> here (byte-checked)
#   surf     surf-stats --per-wrap --ct --ct-only 1 (the fields' geometry only: whole-scroll phase planes OOMed the desk at 137 GB): the on-papyrus share (max CT within +-15 um along the normal
#            >= the scan's Otsu level, scanprep stats)
#   flat     SLIM flatten of the NFLAT (40) wraps with the largest vertices x on-papyrus among those with on-papyrus
#            >= ONPAP (0.5); 4 processes x FTH (6) threads; the largest valid rectangle per wrap
#   render   once the laptop's ink is here (state/ink.ready): rectangles >= MINCM2 (5) upright (tifxyz-scale --flip-rows 1), CT (local) +
#            ift3 ink at 2 px / voxel, CT | ink side-by-side JPEGs sheets/img/<w>_side.jpg, sheets/sheets.tsv sorted by cm2
#   clean    the fields, the flatten inputs, the render tifs (keeps ref, joined, wraps, flat, sheets, ink)
#   CPU stages (ref, surf, flat, render) run at nice 10, wait for MemAvailable >= MEMGB (40) and hold one of CPUSLOTS (2) desk-wide
#   slots (flock C/cpu.slotN), so at most 2 of them run beside the two predicts.
# Start (from a copy, so repo edits never touch a running instance):
#   cd /vesuvius/ufsm/corpus && mkdir -p run && cp src/scripts/corpus/{common,desk}.sh run/ && setsid nohup bash run/desk.sh > desk.out 2>&1 < /dev/null &
# Env: ONLY ("S1 S2"), SPACEGB (150: free desk GB before a predict), GPUS ("0 1"), WAITGPU (1: a lane waits until its GPU holds no other
#   process), STOPF (stop file, default C/STOP).
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
LIST=${ONLY:-$(scrolls)}; TH=${TH:-12}

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
  jq -r '.stats.otsu' /vesuvius/ufsm/u0/prep/$S-$id.json > $O/otsu.txt
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
refq_ready() {   # the fallback: pushed here, forlindesk has > 1 pushed scroll without wraps, and nobody owns this scroll's ref yet
  done_ push || return 1
  local q=$(ssh -n $FD "cd $FDC; n=0; for f in */state/fields.ready; do s=\${f%%/*}; [ -e \$s/state/wraps.ready ] || [ -e \$s/state/failed ] || n=\$((n + 1)); done; echo \$n; test -e $S/state/ref.lock && echo owned" 2>/dev/null)
  case "$q" in *owned*) done_ ref || mark ref "owned on forlindesk"; return 1;; esac; [ "${q:-0}" -gt 1 ]
}
ref() {
  local d=$FDC/$S a b
  ssh -n $FD "mkdir $d/state/ref.lock 2>/dev/null && echo desk > $d/state/ref.lock/owner" || { log "ref: forlindesk took it"; return 0; }
  rm -rf $REF
  if ! /usr/bin/time -v nice -n 10 $U count-field $F $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $REF --model-only 1 --model-stride ${MSTRIDE:-4} \
       --threads $TH --mem-gb ${CFMEM:-24} > $O/logs/ref.log 2>&1 || [ ! -e $REF/model.json ]; then ssh -n $FD "rm -rf $d/state/ref.lock"; return 1; fi   # (forlindesk may take it then)
  rsync -a $REF/ $FD:$d/ref/ || return 1; a=$(bytes $REF); b=$(ssh -n $FD "du -sb --apparent-size $d/ref" | awk '{print $1}')
  [ "$a" = "$b" ] || { log "ref push: $a bytes here, ${b:-?} on forlindesk"; return 1; }
  ssh -n $FD "echo '$(date '+%F %T') desk' > $d/state/ref.ready" && log "ref: computed here, on forlindesk"
}
push() {
  local f=$(ssh -n $FD "df -BG --output=avail $FDC | tail -n 1" | tr -dc 0-9) d=$FDC/$S a b
  [ "${f:-0}" -ge ${FDGB:-250} ] || { log "push: forlindesk has ${f:-?} GB free (FDGB ${FDGB:-250})"; return 1; }
  ssh -n $FD "mkdir -p $d/state $d/logs" && rsync -a $O/box.txt $O/axis.json $FD:$d/ && rsync -a $F/ $FD:$d/fields/ || return 1
  a=$(bytes $F); b=$(ssh -n $FD "du -sb --apparent-size $d/fields" | awk '{print $1}')
  [ "$a" = "$b" ] || { log "push: fields $a bytes here, ${b:-?} on forlindesk"; return 1; }
  ssh -n $FD "echo '$(date '+%F %T') $a field bytes' > $d/state/fields.ready" || return 1
  log "push: fields on forlindesk ($(awk -v b=$a 'BEGIN { printf "%.1f", b / 1e9 }') GB)"
}
wraps_ready() { ssh -n $FD "test -e $FDC/$S/state/wraps.ready"; }
pull() {
  local d=$FDC/$S p a b
  rsync -a $FD:$d/joined/ $O/joined/ && rsync -a $FD:$d/wraps/ $O/wraps/ && rsync -a $FD:$d/logs/ $O/logs/fd/ && rsync -a $FD:$d/slabs.txt $O/ || return 1
  for p in joined wraps; do a=$(bytes $O/$p); b=$(ssh -n $FD "du -sb --apparent-size $d/$p" | awk '{print $1}')
    [ "$a" = "$b" ] || { log "pull: $p $a bytes here, ${b:-?} on forlindesk"; return 1; }; done
  log "pull: joined + $(ls $O/wraps | grep -c '^w') wraps"
}
surf() {
  [ -s $O/otsu.txt ] || jq -r .stats.otsu /vesuvius/ufsm/u0/prep/$S-$(basename $K | cut -c1-14).json > $O/otsu.txt
  N nice -n 5 $U surf-stats $O/wraps --fields $F --model $O/joined --ct-only 1 --threads 16 --per-wrap --ct $R $K/$LV --ct-um $LUM --ct-level $(cat $O/otsu.txt) > $O/logs/surfstats.log 2>&1 || return 1
  log "surf: $(grep -E '^surf-stats: sheet switches' $O/logs/surfstats.log | cut -c1-110) | $(grep -E '^surf-stats: CT: on papyrus' $O/logs/surfstats.log | cut -c1-120)"
}
flat() {   # per-wrap rows of surf-stats --ct: wNNN vertices conf |e|med |e|>.25 theta-sw/cm z-sw/cm on-papyrus
  local list g=0 grp
  list=$(awk -v m=${SWMAX:-2.5} -v p=${ONPAP:-0.5} '$1 ~ /^w[0-9]+$/ && NF >= 8 && $6 <= m && $8 >= p {print $2 * $8, $1}' $O/logs/surfstats.log |
         sort -rg | head -n ${NFLAT:-40} | awk '{print $2}')
  [ -n "$list" ] || { log "flat: no wrap with theta sw/cm <= ${SWMAX:-2.5} and on-papyrus >= ${ONPAP:-0.5}"; mkdir -p $O/flat; : > $O/flat/flatten.log; return 0; }
  echo "$list" > $O/flat-list.txt; rm -rf $O/flat; mkdir -p $O/flat
  N nice -n 5 $U tifxyz-scale $O/wraps $O/flat/orig 1 --wraps $(echo $list | tr ' ' ',') > $O/logs/flat-scale.log 2>&1 || return 1
  for grp in $(echo $list | tr ' ' '\n' | awk '{a[NR % 4] = a[NR % 4] (a[NR % 4] ? "," : "") $1} END {for (k in a) print a[k]}'); do
    N nice -n 5 $U flatten $O/flat/orig $O/flat/flat --wraps $grp --threads ${FTH:-6} > $O/logs/flatten.$g.log 2> $O/logs/flatten.$g.err & g=$((g + 1))
  done; wait
  cat $O/logs/flatten.?.log > $O/flat/flatten.log
  log "flat: $(grep -cE '^w[0-9]+: input' $O/flat/flatten.log) of $(echo $list | wc -w) wraps; rectangles >= ${MINCM2:-5} cm2: $(grep -oE '= [0-9.]+ cm2' $O/flat/flatten.log | awk -v m=${MINCM2:-5} '$2 >= m' | wc -l)"
}
ink_ready() { done_ flat && [ -e $O/state/ink.ready ]; }
render() {
  local P=$O/sheets w cm a50 d st op line il px=$(awk -v u=$LUM 'BEGIN { print u / 2 }')
  il=$(ls $O/ink.zarr | grep -E '^[0-9.]+$' | sort -g | head -n 1); [ -n "$il" ] || return 1
  rm -rf $P; mkdir -p $P/up $P/r $P/img
  printf 'scroll\twrap\trect_cm2\ton_papyrus\taniso_p50\tink_mean\tink_frac50\tside_jpg\n' > $P/sheets.tmp
  grep -E '^w[0-9]+: input' $O/flat/flatten.log | while read -r line; do
    w=${line%%:*}; cm=$(echo "$line" | grep -oE '= [0-9.]+ cm2' | awk '{print $2}'); awk -v c=$cm -v m=${MINCM2:-5} 'BEGIN { exit !(c >= m) }' || continue
    a50=$(grep -E "^$w output grid" $O/flat/flatten.log | grep -oE 'aniso p50/p95/p99 [0-9.]+' | awk '{print $3}')
    op=$(awk -v w=$w '$1 == w {print $8}' $O/logs/surfstats.log)
    N nice -n 5 $U tifxyz-scale $O/flat/flat $P/up 1 --wraps $w --flip-rows 1 > /dev/null 2>&1 || continue
    d=$P/r/$w; mkdir -p $d/in; ln -sfn $P/up/$w $d/in/$w
    UFSM_RENDER_MEM=16 N nice -n 5 $U render $d/in $d --volume $O/ink.zarr $il --ct $R $K/$LV --depth 1 --scale 2 --threads 16 > $d/render.log 2>&1 || continue
    N $U ink-compare $P/img/${w}_side.jpg --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct > $d/stats.txt 2>&1
    N $U ink-compare - --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct --each $P/img/${w}_ --each-ct > /dev/null 2>&1
    st=$(grep -E '^ink ift3:' $d/stats.txt | awk '{print $6"\t"$8}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $S $w $cm ${op:-na} ${a50:-na} "${st:-na	na}" ${w}_side.jpg >> $P/sheets.tmp
  done
  { head -n 1 $P/sheets.tmp; tail -n +2 $P/sheets.tmp | sort -t$'\t' -k3,3gr; } > $P/sheets.tsv; rm -f $P/sheets.tmp
  log "render: $(($(wc -l < $P/sheets.tsv) - 1)) sheets >= ${MINCM2:-5} cm2, $(tail -n +2 $P/sheets.tsv | awk '{s += $3} END {printf "%.1f", s}') cm2, largest $(sed -n 2p $P/sheets.tsv | cut -f2-4)"
}
clean() { rm -rf $F $O/flat/orig $O/sheets/r/*/*.tif; ssh -n $FD "rm -rf $FDC/$S/fields"; return 0; }

# a lane: LANE STAGE NEED FN [G]: every scroll of LIST whose NEED holds (a stage name: its done marker; a function name ending in _ready;
# -: always) and STAGE is not done, claimed; until no scroll is left
cpu_slot() {   # one of CPUSLOTS (2) desk-wide CPU slots for the heavy CPU stages (flock on fd 7)
  local i; while :; do for i in $(seq 1 ${CPUSLOTS:-2}); do exec 7> $C/cpu.slot$i; flock -n 7 && return 0; done; exec 7>&-; sleep 30; done; }
lane() {
  local st=$1 need=$2 fn=$3 left; G=${4:-}
  while ! stopped; do
    left=0
    for S in $LIST; do
      scan $S || continue; failed && continue; done_ $st && continue; left=$((left + 1))
      case $need in -) ;; *_ready) $need || continue;; *) done_ $need || continue;; esac
      claim $st || continue
      case $st in
        fields) while [ "${WAITGPU:-1}" = 1 ] && nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
                while [ $(freegb) -lt ${SPACEGB:-150} ]; do log "fields waits: $(freegb) GB free (SPACEGB ${SPACEGB:-150})"; sleep 600; done;;
        ref|surf|flat|render) memok; cpu_slot;;
      esac
      run $st $fn; unclaim; exec 7>&-   # (fd 7: the CPU slot, if held)
    done
    [ $left = 0 ] && break; sleep 60
  done
}

for S in $LIST; do scan $S; (failed || done_ prep) && continue; ( run prep prep ) &
  while [ $(jobs -r | wc -l) -ge ${PREPPAR:-3} ]; do sleep 5; done; done; wait
for g in ${GPUS:-0 1}; do lane fields prep fields $g & done
lane push fields push &
lane ref refq_ready ref &
lane pull wraps_ready pull &
lane surf pull surf &
lane flat surf flat &
lane render ink_ready render &
lane clean render clean &
wait
ssh -n $FD "touch $FDC/desk.done"
echo "$(date '+%m-%d %H:%M:%S') desk.sh done" >> $C/corpus.log

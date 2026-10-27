#!/usr/bin/env bash
# unroll.sh SCROLL: one scroll of list.tsv through the Paris 4 coarse-to-fine recipe on the desk (forlindesk2, whole machine):
#   prep    box of the scanprep mask (level voxels), axis (list.tsv umbilicus, or ufsm axis = per-z centroid of the masked CT)
#   fields  wind96 fit fields of the box (GPU 0; run3-cf predict settings)  ||  ink: plain ift3-24000 predict of the box (GPU 1)
#   ref     count-field --R 0 --pool 8 --phase-w 4 --model-only (one global LS gauge; CPU)
#   slabs   fit --ref-model REF --ref-snap 1 --uw-repair 0 per 64-cell slab (+16 overlap), full cross-section, lanes on GPUs 0 and 1
#   join    fit-join --ref REF (prints the overlap integer agreement)
#   wraps   ufsm wraps (.sfc, step 4) + surf-stats --per-wrap (GT-free: phase residual switches / cm per wrap)
#   flat    SLIM flatten of the NFLAT largest wraps with theta sw/cm <= SWMAX (largest valid rectangle per wrap)
#   render  flat sheets >= MINCM2: upright (tifxyz-scale --flip-rows 1), CT + ift3 ink at 2 px / voxel, CT | ink side-by-side
#           JPEGs, sheets.tsv (wrap, cm2, aniso, ink mean / frac50), sorted by cm2
#   clean   (CLEAN=1) deletes the fields and the slab fits (the joined model is standalone)
# Done markers $O/state/<stage>.done; a failed stage retries up to RETRY (2) times per invocation, then the scroll is marked failed.
# Root $C/<scroll> (C=/vesuvius/ufsm/corpus). Stop: touch $C/STOP (checked between stages and slabs).
# Env: U (binary), MEMGB (40: MemAvailable GB before fits / count-field), SPACEGB (120: free /vesuvius GB before the predicts),
#   TH (32), NFLAT (40), SWMAX (2.5), MINCM2 (5), CFMEM (48: count-field --mem-gb), INK (1), CLEAN (1).
set -u
H=$(cd "$(dirname "$0")" && pwd); C=${C:-/vesuvius/ufsm/corpus}; S=$1
U=${U:-$C/src/build/ufsm}; TH=${TH:-32}
CK=/vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt; INKCK=/vesuvius/ufsm/ink-ft/runs/ift3/snap/ift3-24000.ckpt
IFS=$'\t' read -r _ R K UM LV AXS < <(awk -F'\t' -v s=$S '$1 == s' $C/list.tsv); [ -n "${K:-}" ] || { echo "$S not in $C/list.tsv"; exit 2; }
O=$C/$S; mkdir -p $O/state $O/logs; cd $O
F=$O/fields; REF=$O/ref; I=64; OV=16
LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }')   # the level's voxel um (fields, wraps, renders)
log() { echo "$(date '+%m-%d %H:%M:%S') $S $*" | tee -a $O/run.log $C/corpus.log; }
done_() { [ -e $O/state/$1.done ]; }
mark() { echo "$(date '+%F %T') ${2:-}" > $O/state/$1.done; }
stopped() { [ -e $C/STOP ]; }
memok() { while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt ${MEMGB:-40} ]; do sleep 30; done; }
space() { local f=$(df -BG --output=avail $C | tail -n 1 | tr -dc 0-9); [ $f -ge ${SPACEGB:-120} ] || { log "only $f GB free (SPACEGB ${SPACEGB:-120})"; return 1; }; }
N() { nice -n 5 "$@"; }
run() {   # run STAGE FN: the stage unless done, with retries
  local st=$1 fn=$2 k t0; done_ $st && return 0; stopped && { log "STOP"; exit 0; }
  for k in $(seq 0 ${RETRY:-2}); do
    t0=$(date +%s); log "$st start (try $k)"
    if $fn; then mark $st "$(( $(date +%s) - t0 )) s"; log "$st done $(( $(date +%s) - t0 )) s"; return 0; fi
    log "$st FAILED (try $k, logs/$st*)"; stopped && exit 0; sleep 30
  done
  touch $O/state/failed; log "giving up at $st"; exit 1
}

# ---- prep: box + axis ----
prep() {
  local id=$(basename $K | cut -c1-14) m mj sh
  m=/vesuvius/ufsm/u0/prep/$S-$id.mask.raw; mj=$m.json; sh=$(jq -r '.shape | join(" ")' $R/$K/$LV/zarr.json)
  read -r GZ GY GX < <(jq -r '.grid | join(" ")' $mj); local cell=$(jq -r .cell $mj)
  # the bounding box of the object cells (code >= 1), padded by one cell, on a 64-voxel grid, clamped to the level's shape
  od -An -v -tu1 -w$((GY * GX)) $m | awk -v Y=$GY -v X=$GX -v c=$cell -v sh="$sh" '
    { for (i = 1; i <= NF; i++) if ($i > 0) { y = int((i - 1) / X); x = (i - 1) % X; z = NR - 1
        if (!n++) { z0 = z1 = z; y0 = y1 = y; x0 = x1 = x }
        if (z < z0) z0 = z; if (z > z1) z1 = z; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x } }
    END { split(sh, s, " "); a[1] = z0; a[2] = y0; a[3] = x0; b[1] = z1; b[2] = y1; b[3] = x1
      for (d = 1; d <= 3; d++) { lo = (a[d] - 1) * c; if (lo < 0) lo = 0; lo = int(lo / 64) * 64; hi = (b[d] + 2) * c; if (hi > s[d]) hi = s[d]; o[d] = lo; e[d] = hi - lo }
      printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], e[1], e[2], e[3] }' > box.txt
  [ -s box.txt ] || return 1
  if [ "$AXS" = auto ]; then N $U axis $R $K axis.json > logs/axis.log 2>&1 || return 1; else cp $AXS axis.json; fi
  log "box $(cat box.txt) (level $LV, $LUM um), axis $AXS"
}
# ---- fields (GPU 0) || ink (GPU 1) ----
fields() {
  space || return 1; rm -rf $F $F.tmp
  N $U predict $CK $R $K $F.tmp --um $UM --level $LV --box $(cat box.txt) --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads 16 --gpu 0 --axis $O/axis.json --levels 1 --fit-zarr $F --fit-h 2 --fit-q 8 --fit-q-phase 0 \
    --no-out 1 > logs/fields.log 2>&1; local rc=$?; rm -rf $F.tmp; [ $rc = 0 ] && [ -e $F/fields.json ]
}
ink() {
  [ "${INK:-1}" = 1 ] || return 0
  N $U predict $INKCK $R $K $O/ink.zarr --um $UM --level $LV --box $(cat box.txt) --shard 1024 --window 1024 --halo 0 --ema 1 --q 32 --threads 16 \
    --gpu 1 --full-grid 1 --resume 1 --occupancy 5 --levels 4 >> logs/ink.log 2>&1
}
# ---- ref ----
ref() {
  rm -rf $REF; memok
  /usr/bin/time -v nice -n 5 $U count-field $F $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $REF --model-only 1 --model-stride 2 \
    --threads $TH --mem-gb ${CFMEM:-48} > logs/ref.log 2>&1 && [ -e $REF/model.json ]
}
# ---- slabs ----
ZE() { jq -r '.shape[0]' $F/fields.json; }
lane() {   # lane GPU: unclaimed slabs in z order
  local g=$1 k s d c0 c1 a b t0 rc ze=$(ZE) ny=$(jq -r '.shape[1]' $F/fields.json) nx=$(jq -r '.shape[2]' $F/fields.json)
  local ns=$(( (ze + I - 1) / I ))
  for k in $(seq 0 $((ns - 1))); do
    stopped && return 0
    s=$(printf 's%03d' $k); d=$O/slabs/$s; mkdir -p $d
    exec 9> $d/claim; flock -n 9 || continue; [ -e $d/done ] && continue
    c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
    rm -rf $d/f $d/fit; bash $H/../wbench/mkcrop.sh $F $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null || continue
    echo "$c0 $c1 $a $b" > $d/cells; memok; t0=$(date +%s)
    env UFSM_FIT_THREADS=12 nice -n 5 $U fit $d/f --gpu $g --out $d/fit --dir 1 --ref-model $REF --ref-snap 1 --uw-repair 0 > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s" >> $O/slabs.txt
    [ $rc = 0 ] && touch $d/done
  done
}
slabs() {
  lane 0 & lane 1 & wait
  local ns=$(( ($(ZE) + I - 1) / I )) nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l)
  log "slabs $nd of $ns done"; [ $nd = $ns ]
}
# ---- join ----
joinst() {
  local args=() d c0 c1 a b
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; N $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}" > logs/join.log 2>&1 || return 1
  log "join: $(grep -oE 'integer agreement[^:]*: [0-9.]+|mean share\(mode\)[^:]*: [0-9.]+' logs/join.log | tr '\n' ' ')"
}
# ---- wraps ----
wraps() {
  rm -rf $O/wraps; N $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH > logs/wraps.log 2>&1 || return 1
  N $U surf-stats $O/wraps --fields $F --threads $TH --per-wrap > logs/surfstats.log 2>&1 || return 1
  log "wraps $(ls $O/wraps | grep -c '^w') | $(grep -E '^surf-stats: sheet switches' logs/surfstats.log | cut -c1-150)"
}
# ---- flatten ----
flat() {
  local list
  list=$(awk -v m=${SWMAX:-2.5} '$1 ~ /^w[0-9]+$/ && NF >= 7 && $6 <= m {print $2, $1}' logs/surfstats.log | sort -rn | head -n ${NFLAT:-40} | awk '{print $2}')
  [ -n "$list" ] || { log "flat: no wrap passes sw/cm <= ${SWMAX:-2.5}"; return 1; }
  echo "$list" > flat-list.txt; rm -rf $O/flat; mkdir -p $O/flat
  N $U tifxyz-scale $O/wraps $O/flat/orig 1 --wraps $(echo $list | tr ' ' ',') > logs/flat-scale.log 2>&1 || return 1
  local g=0 grp; for grp in $(echo $list | tr ' ' '\n' | awk '{a[NR % 4] = a[NR % 4] (a[NR % 4] ? "," : "") $1} END {for (k in a) print a[k]}'); do
    N $U flatten $O/flat/orig $O/flat/flat --wraps $grp --threads $((TH / 4)) > logs/flatten.$g.log 2> logs/flatten.$g.err & g=$((g + 1))
  done; wait
  cat logs/flatten.?.log > $O/flat/flatten.log
  log "flat: $(grep -cE '^w[0-9]+: input' $O/flat/flatten.log) of $(echo $list | wc -w) wraps, rectangles >= ${MINCM2:-5} cm2: $(grep -oE '= [0-9.]+ cm2' $O/flat/flatten.log | awk -v m=${MINCM2:-5} '$2 >= m' | wc -l)"
}
# ---- render ----
inklev() { ls $O/ink.zarr | grep -E '^[0-9.]+$' | sort -g | head -n 1; }
render() {
  local P=$O/sheets w cm a50 d st il=$(inklev) px=$(awk -v u=$LUM 'BEGIN { print u / 2 }')
  rm -rf $P; mkdir -p $P/up $P/r $P/img
  printf 'scroll\twrap\trect_cm2\taniso_p50\tink_mean\tink_frac50\tct_mean\tside_jpg\n' > $P/sheets.tsv.tmp
  grep -E '^w[0-9]+: input' $O/flat/flatten.log | while read -r line; do
    w=${line%%:*}; cm=$(echo "$line" | grep -oE '= [0-9.]+ cm2' | awk '{print $2}'); awk -v c=$cm -v m=${MINCM2:-5} 'BEGIN { exit !(c >= m) }' || continue
    a50=$(grep -E "^$w output grid" $O/flat/flatten.log | grep -oE 'aniso p50/p95/p99 [0-9.]+' | awk '{print $3}')
    N $U tifxyz-scale $O/flat/flat $P/up 1 --wraps $w --flip-rows 1 > /dev/null 2>&1 || continue
    d=$P/r/$w; mkdir -p $d/in; ln -sfn $P/up/$w $d/in/$w
    if [ "${INK:-1}" = 1 ]; then
      UFSM_RENDER_MEM=16 N $U render $d/in $d --volume $O/ink.zarr $il --ct $R $K/$LV --depth 1 --scale 2 --threads $TH > $d/render.log 2>&1 || continue
      N $U ink-compare $P/img/${w}_side.jpg --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct > $d/stats.txt 2>&1
      N $U ink-compare - --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct --each $P/img/${w}_ --each-ct > /dev/null 2>&1
      st=$(grep -E '^ink ift3:' $d/stats.txt | awk '{print $6"\t"$8}')
    else
      UFSM_RENDER_MEM=16 N $U render $d/in $d --volume $R $K/$LV --depth 1 --scale 2 --threads $TH > $d/render.log 2>&1 || continue; st="na	na"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $S $w $cm ${a50:-na} "$st" na ${w}_side.jpg >> $P/sheets.tsv.tmp
  done
  { head -n 1 $P/sheets.tsv.tmp; tail -n +2 $P/sheets.tsv.tmp | sort -t$'\t' -k3,3gr; } > $P/sheets.tsv; rm -f $P/sheets.tsv.tmp
  log "render: $(($(wc -l < $P/sheets.tsv) - 1)) sheets, $(tail -n +2 $P/sheets.tsv | awk '{s += $3} END {printf "%.1f", s}') cm2"
}
clean() {
  [ "${CLEAN:-1}" = 1 ] || return 0
  rm -rf $F $O/slabs/s???/f $O/slabs/s???/fit $O/flat/orig $O/sheets/r/*/*.tif; return 0
}

run prep prep
[ "${PREPONLY:-0}" = 1 ] && exit 0   # (box + axis only: a check of the list)
if ! done_ fields || ! done_ ink; then   # fields on GPU 0 while the ink predicts on GPU 1
  ( run ink ink ) & IP=$!
  run fields fields
  wait $IP || exit 1
fi
run ref ref
run slabs slabs
run join joinst
run wraps wraps
run flat flat
run render render
run clean clean
log "finished"

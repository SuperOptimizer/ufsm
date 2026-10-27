#!/usr/bin/env bash
# fd.sh: the forlindesk side of the corpus unroll (see common.sh). Loop, in list.tsv order:
#   1. renders first: a scroll with flat.done and state/ink.ready (laptop) -> render + clean
#   2. else the next scroll with state/fields.ready (desk: fields + ref) -> slabs -> join -> wraps (+ surf-stats) -> fields deleted -> flat
#   slabs   fit --ref-model REF --ref-snap 1 --uw-repair 0 per 64-cell slab (+16 overlap), full cross-section, one lane per GPU
#   join    fit-join --ref REF (prints the overlap integer agreement)
#   wraps   ufsm wraps (.sfc, step 4) + surf-stats --per-wrap (GT-free: phase-residual switches / cm per wrap)
#   flat    SLIM flatten of the NFLAT largest wraps with theta sw/cm <= SWMAX (largest valid rectangle per wrap), 4 processes
#   render  rectangles >= MINCM2: upright (tifxyz-scale --flip-rows 1), CT (HTTP mirror, chunk cache C/<S>/cache) + ift3 ink at 2 px /
#           voxel, CT | ink side-by-side JPEGs sheets/img/<w>_side.jpg, sheets/sheets.tsv sorted by cm2
#   clean   the slab fits, the flatten inputs, the render tifs, the CT cache (keeps ref, joined, wraps, flat, sheets, ink)
# Ends when the desk has pushed everything (C/desk.done) and nothing is left. Start:
#   cd /vesuvius/ufsm/corpus && mkdir -p run && cp src/scripts/corpus/{common,fd}.sh run/ && setsid nohup bash run/fd.sh > fd.out 2>&1 < /dev/null &
#   (a copy: editing the repo never touches the running loop; fd.sh finds mkcrop.sh in src/scripts/wbench)
# Env: TH (20), MEMGB (12: MemAvailable GB before a fit), NFLAT (40), SWMAX (2.5), MINCM2 (5), CLEAN (1).
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
TH=${TH:-20}; MEMGB=${MEMGB:-12}

lane() {   # lane GPU: unclaimed slabs in z order
  local g=$1 k s d c0 c1 a b t0 rc ze=$(jq -r '.shape[0]' $F/fields.json) ny=$(jq -r '.shape[1]' $F/fields.json) nx=$(jq -r '.shape[2]' $F/fields.json)
  for k in $(seq 0 $(( (ze + I - 1) / I - 1 ))); do
    stopped && return 0
    s=$(printf 's%03d' $k); d=$O/slabs/$s; mkdir -p $d
    exec 9> $d/claim; flock -n 9 || continue; [ -e $d/done ] && continue
    c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
    rm -rf $d/f $d/fit; bash $C/src/scripts/wbench/mkcrop.sh $F $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null || continue
    echo "$c0 $c1 $a $b" > $d/cells; memok; t0=$(date +%s)
    env UFSM_FIT_THREADS=10 nice -n 5 $U fit $d/f --gpu $g --out $d/fit --dir 1 --ref-model $REF --ref-snap 1 --uw-repair 0 > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s" >> $O/slabs.txt
    if [ $rc = 0 ]; then touch $d/done; elif grep -q "no data for the prior" $d/fit.log; then touch $d/empty $d/done; fi   # (air slabs: the box ends)
  done
}
slabs() {
  lane 0 & lane 1 & wait
  local ns=$(( ($(jq -r '.shape[0]' $F/fields.json) + I - 1) / I )) nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l)
  log "slabs $nd of $ns done, $(awk '{s += $NF + 0} END {printf "%.2f", s / 3600}' $O/slabs.txt) GPU-h"; [ $nd = $ns ]
}
joinst() {
  local args=() d c0 c1 a b
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; [ -e $d/empty ] && continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; N $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}" > $O/logs/join.log 2>&1 || return 1
  log "join: $(grep -oE 'integer agreement[^:]*: [0-9.]+|mean share\(mode\)[^:]*: [0-9.]+' $O/logs/join.log | tr '\n' ' ')"
}
wraps() {
  rm -rf $O/wraps; N $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH > $O/logs/wraps.log 2>&1 || return 1
  N $U surf-stats $O/wraps --fields $F --threads $TH --per-wrap > $O/logs/surfstats.log 2>&1 || return 1
  log "wraps $(ls $O/wraps | grep -c '^w') | $(grep -E '^surf-stats: sheet switches' $O/logs/surfstats.log | cut -c1-150)"
  [ "${CLEAN:-1}" = 1 ] && rm -rf $F $O/slabs/s???/f $O/slabs/s???/fit; return 0   # (the joined model is standalone)
}
flat() {
  local list g=0 grp
  list=$(awk -v m=${SWMAX:-2.5} '$1 ~ /^w[0-9]+$/ && NF >= 7 && $6 <= m {print $2, $1}' $O/logs/surfstats.log | sort -rn | head -n ${NFLAT:-40} | awk '{print $2}')
  [ -n "$list" ] || { log "flat: no wrap with theta sw/cm <= ${SWMAX:-2.5}"; return 1; }
  echo "$list" > $O/flat-list.txt; rm -rf $O/flat; mkdir -p $O/flat
  N $U tifxyz-scale $O/wraps $O/flat/orig 1 --wraps $(echo $list | tr ' ' ',') > $O/logs/flat-scale.log 2>&1 || return 1
  for grp in $(echo $list | tr ' ' '\n' | awk '{a[NR % 4] = a[NR % 4] (a[NR % 4] ? "," : "") $1} END {for (k in a) print a[k]}'); do
    N $U flatten $O/flat/orig $O/flat/flat --wraps $grp --threads $((TH / 4)) > $O/logs/flatten.$g.log 2> $O/logs/flatten.$g.err & g=$((g + 1))
  done; wait
  cat $O/logs/flatten.?.log > $O/flat/flatten.log
  log "flat: $(grep -cE '^w[0-9]+: input' $O/flat/flatten.log) of $(echo $list | wc -w) wraps; rectangles >= ${MINCM2:-5} cm2: $(grep -oE '= [0-9.]+ cm2' $O/flat/flatten.log | awk -v m=${MINCM2:-5} '$2 >= m' | wc -l)"
}
render() {
  local P=$O/sheets w cm a50 d st line il px=$(awk -v u=$LUM 'BEGIN { print u / 2 }')
  il=$(ls $O/ink.zarr | grep -E '^[0-9.]+$' | sort -g | head -n 1); [ -n "$il" ] || return 1
  rm -rf $P; mkdir -p $P/up $P/r $P/img
  printf 'scroll\twrap\trect_cm2\taniso_p50\tink_mean\tink_frac50\tside_jpg\n' > $P/sheets.tmp
  grep -E '^w[0-9]+: input' $O/flat/flatten.log | while read -r line; do
    w=${line%%:*}; cm=$(echo "$line" | grep -oE '= [0-9.]+ cm2' | awk '{print $2}'); awk -v c=$cm -v m=${MINCM2:-5} 'BEGIN { exit !(c >= m) }' || continue
    a50=$(grep -E "^$w output grid" $O/flat/flatten.log | grep -oE 'aniso p50/p95/p99 [0-9.]+' | awk '{print $3}')
    N $U tifxyz-scale $O/flat/flat $P/up 1 --wraps $w --flip-rows 1 > /dev/null 2>&1 || continue
    d=$P/r/$w; mkdir -p $d/in; ln -sfn $P/up/$w $d/in/$w
    UFSM_RENDER_MEM=16 N $U render $d/in $d --volume $O/ink.zarr $il --ct $MIRROR $MK/$LV --cache $O/cache --depth 1 --scale 2 --threads $TH > $d/render.log 2>&1 || continue
    N $U ink-compare $P/img/${w}_side.jpg --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct > $d/stats.txt 2>&1
    N $U ink-compare - --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct --each $P/img/${w}_ --each-ct > /dev/null 2>&1
    st=$(grep -E '^ink ift3:' $d/stats.txt | awk '{print $6"\t"$8}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $S $w $cm ${a50:-na} "${st:-na	na}" ${w}_side.jpg >> $P/sheets.tmp
  done
  { head -n 1 $P/sheets.tmp; tail -n +2 $P/sheets.tmp | sort -t$'\t' -k3,3gr; } > $P/sheets.tsv; rm -f $P/sheets.tmp
  log "render: $(($(wc -l < $P/sheets.tsv) - 1)) sheets >= ${MINCM2:-5} cm2, $(tail -n +2 $P/sheets.tsv | awk '{s += $3} END {printf "%.1f", s}') cm2, largest $(sed -n 2p $P/sheets.tsv | cut -f2,3)"
}
clean() { [ "${CLEAN:-1}" = 1 ] || return 0; rm -rf $F $O/slabs/s???/f $O/slabs/s???/fit $O/flat/orig $O/sheets/r/*/*.tif $O/cache; return 0; }

while ! stopped; do
  rs=; fs=; left=0
  for S in $(scrolls); do
    scan $S || continue; failed && continue; done_ clean && continue; left=$((left + 1))
    [ -z "$rs" ] && done_ flat && [ -e $O/state/ink.ready ] && rs=$S
    [ -z "$fs" ] && ! done_ flat && [ -e $O/state/fields.ready ] && fs=$S
  done
  if [ -n "$rs" ]; then S=$rs; scan $S; run render render && run clean clean && log "finished"; continue; fi
  if [ -n "$fs" ]; then S=$fs; scan $S; run slabs slabs && run join joinst && run wraps wraps && run flat flat; continue; fi
  [ -e $C/desk.done ] && [ $left = 0 ] && break
  sleep 300
done
echo "$(date '+%m-%d %H:%M:%S') fd.sh done" >> $C/corpus.log

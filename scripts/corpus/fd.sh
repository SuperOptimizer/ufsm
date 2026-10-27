#!/usr/bin/env bash
# fd.sh: the forlindesk side of the corpus unroll (see common.sh): every scroll whose state/ready the desk wrote, in list.tsv order:
#   ref     count-field --R 0 --pool 8 --phase-w 4 --model-only --model-stride 2 (one global LS gauge; CPU, runs alone)
#   slabs   fit --ref-model REF --ref-snap 1 --uw-repair 0 per 64-cell slab (+16 overlap), full cross-section, one lane per GPU
#   join    fit-join --ref REF (prints the overlap integer agreement)
#   wraps   ufsm wraps (.sfc, step 4) + surf-stats --per-wrap (GT-free: phase-residual switches / cm per wrap)
#   flat    SLIM flatten of the NFLAT largest wraps with theta sw/cm <= SWMAX (largest valid rectangle per wrap)
#   render  rectangles >= MINCM2: upright (tifxyz-scale --flip-rows 1), CT + ift3 ink at 2 px / voxel, CT | ink side-by-side JPEGs,
#           sheets/sheets.tsv sorted by cm2
#   clean   deletes the fields, the slab fits and the pushed CT (keeps ref, joined, wraps, flat, sheets, ink)
# Waits (polls every 5 min) for the next ready scroll; ends when desk.sh has finished (C/desk.done, written by the desk push of the last scroll) and nothing is left.
# Start: cd /vesuvius/ufsm/corpus && setsid nohup bash src/scripts/corpus/fd.sh > fd.out 2>&1 < /dev/null &
# Env: TH (20), CFMEM (24: count-field --mem-gb), MEMGB (12: MemAvailable GB before a fit), NFLAT (40), SWMAX (2.5), MINCM2 (5), CLEAN (1).
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
TH=${TH:-20}; MEMGB=${MEMGB:-12}

ref() {
  rm -rf $REF; memok
  /usr/bin/time -v nice -n 5 $U count-field $F $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $REF --model-only 1 --model-stride 2 \
    --threads $TH --mem-gb ${CFMEM:-24} > $O/logs/ref.log 2>&1 && [ -e $REF/model.json ]
}
lane() {   # lane GPU: unclaimed slabs in z order
  local g=$1 k s d c0 c1 a b t0 rc ze=$(jq -r '.shape[0]' $F/fields.json) ny=$(jq -r '.shape[1]' $F/fields.json) nx=$(jq -r '.shape[2]' $F/fields.json)
  for k in $(seq 0 $(( (ze + I - 1) / I - 1 ))); do
    stopped && return 0
    s=$(printf 's%03d' $k); d=$O/slabs/$s; mkdir -p $d
    exec 9> $d/claim; flock -n 9 || continue; [ -e $d/done ] && continue
    c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
    rm -rf $d/f $d/fit; bash $H/../wbench/mkcrop.sh $F $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null || continue
    echo "$c0 $c1 $a $b" > $d/cells; memok; t0=$(date +%s)
    env UFSM_FIT_THREADS=10 nice -n 5 $U fit $d/f --gpu $g --out $d/fit --dir 1 --ref-model $REF --ref-snap 1 --uw-repair 0 > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s" >> $O/slabs.txt
    [ $rc = 0 ] && touch $d/done
  done
}
slabs() {
  lane 0 & lane 1 & wait
  local ns=$(( ($(jq -r '.shape[0]' $F/fields.json) + I - 1) / I )) nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l)
  log "slabs $nd of $ns done, $(awk '{s += $NF + 0} END {printf "%.2f", s / 3600}' $O/slabs.txt) GPU-h"; [ $nd = $ns ]
}
joinst() {
  local args=() d c0 c1 a b
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; N $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}" > $O/logs/join.log 2>&1 || return 1
  log "join: $(grep -oE 'integer agreement[^:]*: [0-9.]+|mean share\(mode\)[^:]*: [0-9.]+' $O/logs/join.log | tr '\n' ' ')"
}
wraps() {
  rm -rf $O/wraps; N $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH > $O/logs/wraps.log 2>&1 || return 1
  N $U surf-stats $O/wraps --fields $F --threads $TH --per-wrap > $O/logs/surfstats.log 2>&1 || return 1
  log "wraps $(ls $O/wraps | grep -c '^w') | $(grep -E '^surf-stats: sheet switches' $O/logs/surfstats.log | cut -c1-150)"
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
  local P=$O/sheets w cm a50 d st line il cn=$(basename $K) px=$(awk -v u=$LUM 'BEGIN { print u / 2 }')
  il=$(ls $O/ink.zarr 2>/dev/null | grep -E '^[0-9.]+$' | sort -g | head -n 1)
  rm -rf $P; mkdir -p $P/up $P/r $P/img
  printf 'scroll\twrap\trect_cm2\taniso_p50\tink_mean\tink_frac50\tside_jpg\n' > $P/sheets.tmp
  grep -E '^w[0-9]+: input' $O/flat/flatten.log | while read -r line; do
    w=${line%%:*}; cm=$(echo "$line" | grep -oE '= [0-9.]+ cm2' | awk '{print $2}'); awk -v c=$cm -v m=${MINCM2:-5} 'BEGIN { exit !(c >= m) }' || continue
    a50=$(grep -E "^$w output grid" $O/flat/flatten.log | grep -oE 'aniso p50/p95/p99 [0-9.]+' | awk '{print $3}')
    N $U tifxyz-scale $O/flat/flat $P/up 1 --wraps $w --flip-rows 1 > /dev/null 2>&1 || continue
    d=$P/r/$w; mkdir -p $d/in; ln -sfn $P/up/$w $d/in/$w
    if [ -n "$il" ]; then
      UFSM_RENDER_MEM=16 N $U render $d/in $d --volume $O/ink.zarr $il --ct $O/ct $cn/$LV --depth 1 --scale 2 --threads $TH > $d/render.log 2>&1 || continue
      N $U ink-compare $P/img/${w}_side.jpg --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct > $d/stats.txt 2>&1
      N $U ink-compare - --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $px --stretch --stretch-ct --each $P/img/${w}_ --each-ct > /dev/null 2>&1
      st=$(grep -E '^ink ift3:' $d/stats.txt | awk '{print $6"\t"$8}')
    else
      UFSM_RENDER_MEM=16 N $U render $d/in $d --volume $O/ct $cn/$LV --depth 1 --scale 2 --threads $TH > $d/render.log 2>&1 || continue; st="na	na"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $S $w $cm ${a50:-na} "$st" ${w}_side.jpg >> $P/sheets.tmp
  done
  { head -n 1 $P/sheets.tmp; tail -n +2 $P/sheets.tmp | sort -t$'\t' -k3,3gr; } > $P/sheets.tsv; rm -f $P/sheets.tmp
  log "render: $(($(wc -l < $P/sheets.tsv) - 1)) sheets >= ${MINCM2:-5} cm2, $(tail -n +2 $P/sheets.tsv | awk '{s += $3} END {printf "%.1f", s}') cm2, largest $(sed -n 2p $P/sheets.tsv | cut -f2,3)"
}
clean() { [ "${CLEAN:-1}" = 1 ] || return 0; rm -rf $F $O/slabs/s???/f $O/slabs/s???/fit $O/flat/orig $O/sheets/r/*/*.tif $O/ct; return 0; }
unroll() { run ref ref && run slabs slabs && run join joinst && run wraps wraps && run flat flat && run render render && run clean clean && log "finished"; }

while ! stopped; do
  next=; left=0
  for S in $(tail -n +2 $C/list.tsv | cut -f1); do
    scan $S || continue; done_ clean && continue; [ -e $O/state/failed ] && continue
    left=$((left + 1)); [ -e $O/state/ready ] && { next=$S; break; }
  done
  if [ -n "$next" ]; then S=$next; scan $S; unroll; continue; fi
  [ -e $C/desk.done ] && break   # (the desk has pushed everything it will push)
  sleep 300
done
echo "$(date '+%m-%d %H:%M:%S') fd.sh done" >> $C/corpus.log

#!/bin/bash
# strip_stream.sh OUT: flatten an existing OUT/strip (ufsm strip), render its stripes and mosaic them with bounded disk: three middle
# stripes are rendered first for the contrast stretch (strip-mosaic --stretch-only), then strip-mosaic runs with --stretch
# --wait-renders --rm-renders while the stripes render in order, at most MAXR render pairs on disk at once. Same outputs as strip_run.sh
# (img/: stripe JPEGs, overview.jpg, strip_long.jpg, papyrus tables), progress.log lines "strip-flatten done", "mosaic done".
#   env: U, CTR CTK (CT root / key at 9.6 um), INK INKK, DECIM, NT, RENDER_NT, MOS_NT, MAXR, CTL, FLATX; STOP file: OUT/STOP
set -u
OUT=$1
U=${U:-/vesuvius/ufsm/scroll1/src/build/ufsm}
CTR=${CTR:?CT root}; CTK=${CTK:?CT key}; INK=${INK:?ink zarr}; INKK=${INKK:-9.6}
DECIM=${DECIM:-4}; NT=${NT:-12}; RENDER_NT=${RENDER_NT:-8}; MOS_NT=${MOS_NT:-6}; MAXR=${MAXR:-10}; CTL=${CTL:-80}; FLATX=${FLATX:-}
mkdir -p $OUT/logs $OUT/render
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/progress.log; }
T0=$(date +%s)
[ -e $OUT/strip/strip.sfc ] || { log "no strip"; exit 1; }
if [ ! -e $OUT/strip/flat/flat.json ]; then
  log "strip-flatten start"
  nice -n 10 $U strip-flatten $OUT/strip --decim $DECIM --threads $NT --flip-rows 1 --cg 60 --tol 3e-4 $FLATX > $OUT/logs/flatten.log 2>&1 || { log "strip-flatten FAILED"; exit 1; }
  log "strip-flatten done: $(grep -E '^strip-flatten: (strip|flattened|[0-9]+ stripes)' $OUT/logs/flatten.log | tr '\n' ' ')"
fi
S=($(ls $OUT/strip/flat/stripes)); N=${#S[@]}
rend() {   # one stripe -> render/<s>.tif, <s>_ct.tif, <s>.done
  local s=$1; [ -e $OUT/render/$s.done ] && return 0
  mkdir -p $OUT/render/in_$s; ln -sfn $OUT/strip/flat/stripes/$s $OUT/render/in_$s/$s
  UFSM_RENDER_MEM=16 nice -n 10 $U render $OUT/render/in_$s $OUT/render --volume $INK $INKK --ct $CTR $CTK --depth 1 --scale 1 --threads $RENDER_NT > $OUT/logs/render_$s.log 2>&1 || { log "render $s FAILED"; return 1; }
  rm -r $OUT/render/in_$s; touch $OUT/render/$s.done; log "render $s done"
}
if [ -e $OUT/stretch.txt ]; then ST=$(cat $OUT/stretch.txt); else
  for k in $((N / 2 - 1)) $((N / 2)) $((N / 2 + 1)); do [ $k -ge 0 ] && [ $k -lt $N ] && { rend ${S[$k]} || exit 1; }; done
  ST=$(nice -n 10 $U strip-mosaic $OUT/strip/flat $OUT/render $OUT/img --stretch-only 1 --threads 3 2> $OUT/logs/stretch.log | awk '/^stretch/ {print $2}')
  [ -n "$ST" ] || { log "stretch FAILED"; exit 1; }
  echo $ST > $OUT/stretch.txt; log "stretch $ST (from ${S[$((N / 2))]} and its neighbours)"
fi
nice -n 10 $U strip-mosaic $OUT/strip/flat $OUT/render $OUT/img --ct-level $CTL --threads $MOS_NT --stretch $ST --wait-renders 1 --rm-renders 1 > $OUT/logs/mosaic.log 2>&1 &
MP=$!
for s in "${S[@]}"; do
  [ -e $OUT/STOP ] && { kill $MP; exit 0; }
  while [ $(ls $OUT/render | grep -c '_ct\.tif$') -ge $MAXR ] && kill -0 $MP 2> /dev/null; do sleep 10; done
  kill -0 $MP 2> /dev/null || break
  rend $s || { kill $MP; exit 1; }
done
wait $MP || { log "mosaic FAILED"; exit 1; }
log "mosaic done: $(grep '^strip-mosaic:' $OUT/logs/mosaic.log | tail -1)"
log "flatten + render + mosaic in $(( $(date +%s) - T0 )) s"

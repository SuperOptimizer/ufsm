#!/bin/bash
# strip_run.sh OUT WRAPS [strip flags]: a whole scroll (or a wrap range) as ONE continuous surface, flattened as one strip, rendered
# (CT + ink) as stripes, mosaicked into stripe JPEGs + overview + on-papyrus tables. Stages skip when their outputs exist.
#   env: U (ufsm binary), CTR CTK (CT root/key, 9.6 um), INK INKK (ink root/key), DECIM (lattice every D vertices), NT (threads),
#        FLATX (extra strip-flatten flags), RENDER_NT, CTL (on-papyrus CT level), STOP file: $OUT/STOP
set -u
OUT=$1; WR=$2; shift 2
U=${U:-/vesuvius/ufsm/scroll1/src/build/ufsm}
CTR=${CTR:-/vesuvius/ufsm/volcomp}; CTK=${CTK:-PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2}
INK=${INK:-/vesuvius/ufsm/paris4/ink/ift3-24000-ov8.zarr}; INKK=${INKK:-9.6}
DECIM=${DECIM:-4}; NT=${NT:-12}; RENDER_NT=${RENDER_NT:-8}; CTL=${CTL:-80}; FLATX=${FLATX:-}
mkdir -p $OUT/logs
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/progress.log; }
T0=$(date +%s)
if [ ! -e $OUT/strip/strip.sfc ]; then
  log "strip start"
  nice -n 10 $U strip $WR $OUT/strip --threads $NT "$@" > $OUT/logs/strip.log 2>&1 || { log "strip FAILED"; exit 1; }
  log "strip done: $(grep '^strip:' $OUT/logs/strip.log | tail -1)"
fi
[ -e $OUT/STOP ] && exit 0
if [ ! -e $OUT/strip/flat/flat.json ]; then
  log "strip-flatten start"
  nice -n 10 $U strip-flatten $OUT/strip --decim $DECIM --threads $NT --flip-rows 1 --cg 60 --tol 3e-4 $FLATX > $OUT/logs/flatten.log 2>&1 || { log "strip-flatten FAILED"; exit 1; }
  log "strip-flatten done: $(grep -E '^strip-flatten: (strip|flattened|[0-9]+ stripes)' $OUT/logs/flatten.log | tr '\n' ' ')"
fi
[ -e $OUT/STOP ] && exit 0
mkdir -p $OUT/render
for s in $(ls $OUT/strip/flat/stripes); do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/render/${s}_ct.tif ] && continue
  mkdir -p $OUT/render/in_$s; ln -sfn $OUT/strip/flat/stripes/$s $OUT/render/in_$s/$s
  UFSM_RENDER_MEM=16 nice -n 10 $U render $OUT/render/in_$s $OUT/render --volume $INK $INKK --ct $CTR $CTK --depth 1 --scale 1 --threads $RENDER_NT > $OUT/logs/render_$s.log 2>&1 || { log "render $s FAILED"; exit 1; }
  rm -r $OUT/render/in_$s
  log "render $s done"
done
nice -n 10 $U strip-mosaic $OUT/strip/flat $OUT/render $OUT/img --ct-level $CTL > $OUT/logs/mosaic.log 2>&1 || { log "mosaic FAILED"; exit 1; }
log "mosaic done: $(grep '^strip-mosaic:' $OUT/logs/mosaic.log | tail -1)"
log "all done in $(( $(date +%s) - T0 )) s"

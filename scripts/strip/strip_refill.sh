#!/bin/bash
# strip_refill.sh OUT: re-render every stripe of a finished strip (OUT/strip/flat/stripes) with hole filling and mosaic them into
# OUT/img_fill, bounded disk like strip_stream.sh (strip-mosaic --wait-renders --rm-renders, at most MAXR render pairs on disk).
# Renders go to OUT/render_fill (removed by the mosaic after use), the per-pixel fill flags (<s>_fill.tif: 0 on the surface, k / 50+k
# filled from +-k voxels along the normal (+100 in an interpolated grid hole), 250 inpainted, 254 no surface, 255 still empty) are kept in OUT/fillmap with the render
# logs' fill shares in OUT/logs/refill_<s>.log. The contrast stretch is OUT/stretch.txt (the original run's, so both mosaics compare)
# or STRETCH. Resumable: rendered stripes still on disk are reused after a stop; stripes the stopped mosaic already consumed (and
# deleted) are rendered again, since the restarted mosaic needs every stripe.
# PAPMOS=DIR: a second mosaic in the papyrus look (strip-mosaic --papyrus PGAIN --papyrus-bg 128 --overview-px PAPPX) beside img_fill;
# both mosaics read each render, which is deleted once both logged it (PGAIN default 1, PAPPX 8000).
# PAP=DIR: also a papyrus-look preview per stripe (ink-compare --papyrus, --max-px 8000) DIR/<s>.jpg, made before the mosaic takes the
# render. MEMGB: each render waits until MemAvailable exceeds it (default 80: a stripe render is ~40 GB RSS, keep 40 GB for the rest).
#   env: U, CTR CTK (CT root / key at 9.6 um), INK INKK, RENDER_NT, MOS_NT, MAXR, CTL, FILLX (render fill flags), STRETCH, PAP, MEMGB;
#   STOP: OUT/STOP_FILL
set -u
OUT=$1
U=${U:-/vesuvius/ufsm/scroll1/src/build/ufsm}
CTR=${CTR:?CT root}; CTK=${CTK:?CT key}; INK=${INK:?ink zarr}; INKK=${INKK:-9.6}
RENDER_NT=${RENDER_NT:-8}; MOS_NT=${MOS_NT:-6}; MAXR=${MAXR:-6}; CTL=${CTL:-80}; PAP=${PAP:-}; MEMGB=${MEMGB:-80}; PAPMOS=${PAPMOS:-}; PGAIN=${PGAIN:-1}; PAPPX=${PAPPX:-8000}
FILLX=${FILLX:---fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 1}
R=$OUT/render_fill; I=$OUT/img_fill; M=$OUT/fillmap
mkdir -p $OUT/logs $R $M
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/progress_fill.log; }
T0=$(date +%s)
[ -e $OUT/strip/flat/flat.json ] || { log "no flattened strip"; exit 1; }
ST=${STRETCH:-$(cat $OUT/stretch.txt 2> /dev/null)}; [ -n "$ST" ] || { log "no stretch (OUT/stretch.txt or STRETCH)"; exit 1; }
S=($(ls $OUT/strip/flat/stripes))
rend() {   # one stripe -> render_fill/<s>.tif, <s>_ct.tif, <s>.done; fillmap/<s>_fill.tif
  local s=$1; [ -e $R/$s.done ] && return 0
  while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt $MEMGB ]; do sleep 30; done
  mkdir -p $R/in_$s; ln -sfn $OUT/strip/flat/stripes/$s $R/in_$s/$s
  UFSM_RENDER_MEM=16 nice -n 10 $U render $R/in_$s $R --volume $INK $INKK --ct $CTR $CTK --depth 1 --scale 1 --threads $RENDER_NT $FILLX > $OUT/logs/refill_$s.log 2>&1 || { log "render $s FAILED"; return 1; }
  [ -e $R/${s}_fill.tif ] && mv $R/${s}_fill.tif $M/
  if [ -n "$PAP" ]; then mkdir -p $PAP
    nice -n 10 $U ink-compare $PAP/$s.jpg --ct $R/${s}_ct.tif --ink ink=$R/$s.tif --papyrus ink --papyrus-gain 1 --papyrus-bg 128 --stretch --stretch-ct \
      --px-um 9.6 --max-px 8000 --threads $RENDER_NT > $OUT/logs/refill_pap_$s.log 2>&1 || log "preview $s failed (exit $?)"; fi
  rm -r $R/in_$s; touch $R/$s.done; log "render $s done: $(grep -o 'void on the surface.*' $OUT/logs/refill_$s.log | head -1)"
}
for d in $R/*.done; do [ -e "$d" ] || continue; s=$(basename $d .done); [ -e $R/${s}_ct.tif ] || rm -f $d; done   # the mosaic needs every stripe: re-render those it already consumed
if [ -z "$PAPMOS" ]; then
  nice -n 10 $U strip-mosaic $OUT/strip/flat $R $I --ct-level $CTL --threads $MOS_NT --stretch $ST --wait-renders 1 --rm-renders 1 --fill-maps $M > $OUT/logs/mosaic_fill.log 2>&1 &
  MPS=($!)
else   # two mosaics on the same renders: the script deletes a render once both have logged it
  nice -n 10 $U strip-mosaic $OUT/strip/flat $R $I --ct-level $CTL --threads $MOS_NT --stretch $ST --wait-renders 1 --fill-maps $M > $OUT/logs/mosaic_fill.log 2>&1 &
  MPS=($!)
  nice -n 10 $U strip-mosaic $OUT/strip/flat $R $PAPMOS --ct-level $CTL --threads $MOS_NT --stretch $ST --wait-renders 1 --papyrus $PGAIN --papyrus-bg 128 --overview-px $PAPPX --fill-maps $M > $OUT/logs/mosaic_fill_pap.log 2>&1 &
  MPS+=($!)
fi
clean() {   # PAPMOS: renders both mosaics have taken
  [ -n "$PAPMOS" ] || return 0
  local d s; for d in $R/*.done; do [ -e "$d" ] || continue; s=$(basename $d .done); [ -e $R/${s}_ct.tif ] || continue
    grep -q "^strip-mosaic: $s .* done" $OUT/logs/mosaic_fill.log && grep -q "^strip-mosaic: $s .* done" $OUT/logs/mosaic_fill_pap.log && rm -f $R/${s}_ct.tif $R/$s.tif; done
}
alive() { local p; for p in "${MPS[@]}"; do kill -0 $p 2> /dev/null || return 1; done; return 0; }
for s in "${S[@]}"; do
  [ -e $OUT/STOP_FILL ] && { kill "${MPS[@]}"; exit 0; }
  clean
  while [ $(ls $R | grep -c '_ct\.tif$') -ge $MAXR ] && alive; do sleep 10; clean; done
  alive || break
  rend $s || { kill "${MPS[@]}"; exit 1; }
done
for p in "${MPS[@]}"; do while kill -0 $p 2> /dev/null; do sleep 10; clean; done; wait $p || { log "mosaic FAILED"; exit 1; }; done
clean
log "mosaic done: $(grep '^strip-mosaic:' $OUT/logs/mosaic_fill.log | tail -1)"
[ -n "$PAPMOS" ] && log "papyrus mosaic done: $(grep '^strip-mosaic:' $OUT/logs/mosaic_fill_pap.log | tail -1)"
log "refill render + mosaic in $(( $(date +%s) - T0 )) s"

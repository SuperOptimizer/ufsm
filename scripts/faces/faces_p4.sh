#!/bin/bash
# faces_p4.sh: recto / verso faces of Paris 4 wraps (ufsm faces) on the p4full flattened grids, each face rendered with CT + ift3 ink on
# the SAME flat uv (pixel-aligned), then per wrap: 3-panel JPEG (CT at recto | ift3 recto | ift3 verso, 9.6 um/px) + 8000 px preview, and
# the recto/verso ink agreement RGB (recto magenta, verso green, both white) full + 8000 px preview. Render tifs deleted after the JPEGs.
# Env: WRAPS (default w191..w207), O (output dir), U (binary), FLAT (flattened wraps: tifxyz dirs), MINMEM (GB MemAvailable gate). STOP: $O/STOP.
O=${O:-/vesuvius/ufsm/faces/p4}; U=${U:-/vesuvius/ufsm/faces/bin/ufsm}; FLAT=${FLAT:-/vesuvius/ufsm/faces/flat}
R=/vesuvius/ufsm/volcomp; K=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2
I=/vesuvius/ufsm/paris4/ink/ift3-24000-ov8.zarr; UMB=/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json
GT=/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr
WRAPS=${WRAPS:-$(seq -s, -f 'w%03g' 191 207)}; MINMEM=${MINMEM:-40}; TH=${TH:-10}
mkdir -p $O/out $O/up $O/r $O/img $O/prev $O/logs
log() { echo "$(date '+%F %T') $*" >> $O/progress.log; }
if [ ! -e $O/faces.done ]; then
  log "faces start: $WRAPS"
  nice -n 5 $U faces $FLAT $O/out --wraps $WRAPS --ct $R $K --ct-um 9.6 --umb $UMB --umb-um 2.4 --gt $GT 9.6 --gt-um 9.6 --threads $TH \
    > $O/faces.tsv 2> $O/logs/faces.log || { log "faces FAILED"; exit 1; }
  touch $O/faces.done; log "faces done"
fi
memok() { [ $(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo) -ge $MINMEM ]; }
for w in ${WRAPS//,/ }; do
  [ -e $O/STOP ] && break
  [ -e $O/img/${w}_rv_rgb.jpg ] && continue
  while ! memok; do sleep 30; done
  d=$O/r/$w; mkdir -p $d/in
  nice -n 5 $U tifxyz-scale $O/out $O/up 1 --wraps ${w}_recto,${w}_verso --flip-rows 1 > /dev/null 2>&1 || { log "$w upscale failed"; continue; }
  ln -sfn $O/up/${w}_recto $d/in/${w}_recto; ln -sfn $O/up/${w}_verso $d/in/${w}_verso
  UFSM_RENDER_MEM=16 nice -n 5 $U render $d/in $d --volume $I 9.6 --ct $R $K --depth 1 --scale 2 --threads $TH > $d/render.log 2>&1 || { log "$w render failed"; continue; }
  C="--ct $d/${w}_recto_ct.tif --px-um 4.8 --stretch --stretch-ct --threads $TH"
  nice -n 5 $U ink-compare $O/img/${w}_rv_side.jpg $C --ink "recto=$d/${w}_recto.tif" --ink "verso=$d/${w}_verso.tif" --max-px 65000 > $d/stats.txt 2>&1
  nice -n 5 $U ink-compare $O/prev/${w}_rv_side.jpg $C --ink "recto=$d/${w}_recto.tif" --ink "verso=$d/${w}_verso.tif" --max-px 8000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/prev/${w}_rv_rgb.jpg $C --ink "recto=$d/${w}_recto.tif" --ink "verso=$d/${w}_verso.tif" --rgb-agree recto,verso --max-px 8000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/img/${w}_rv_rgb.jpg $C --ink "recto=$d/${w}_recto.tif" --ink "verso=$d/${w}_verso.tif" --rgb-agree recto,verso --max-px 65000 > /dev/null 2>&1
  [ -e $O/img/${w}_rv_rgb.jpg ] && rm -f $d/*.tif
  log "$w done: $(grep -E '^corr' $d/stats.txt)"
done
log "renders done: $(ls $O/img/*_rv_rgb.jpg 2>/dev/null | wc -l) wraps"

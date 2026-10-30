#!/bin/bash
# faces_p4c.sh (faces_p4b.sh + viridis outputs): Paris 4 recto / verso faces, whole wraps. Producer: ufsm faces per wrap (text wraps w191-w207 first, then every wrap
# p4full has flattened, largest first) on p4full's flat grids (read only). Consumer (SPAR renders at a time, MemAvailable gate): both
# faces rendered on the same flat uv (CT + ift3 ink, --depth 0: each face reads its own voxels, no +-1 voxel window across the ~4 voxel
# sheet), then <w>_rv_rgb.jpg (recto magenta, verso green, both white; 9.6 um/px) + 8000 px preview, and <w>_rv_side.jpg (CT | recto |
# verso) + preview, and each face's ink through viridis (<w>_recto_viridis.jpg, <w>_verso_viridis.jpg full + 8000 px previews, prev/<w>_rv_viridis.jpg
# recto | verso). Render tifs deleted after the JPEGs. New wraps first, then wraps rendered before the viridis outputs existed. STOP: $O/STOP. WAITPID: a faces run already writing $O/out (its table: faces.tsv).
O=${O:-/vesuvius/ufsm/faces/p4}; U=${U:-/vesuvius/ufsm/faces/bin/ufsm}; P4=/vesuvius/ufsm/p4full
R=/vesuvius/ufsm/volcomp; K=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/2
I=/vesuvius/ufsm/paris4/ink/ift3-24000-ov8.zarr; UMB=/vesuvius/ufsm/umbilicus/PHercParis4/umbilicus-full-resolution.json
GT=/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr
SPAR=${SPAR:-2}; MINMEM=${MINMEM:-40}; TH=${TH:-8}; FTH=${FTH:-10}
TEXT=$(seq -f 'w%03g' 191 207)
mkdir -p $O/out $O/up $O/r $O/img $O/prev $O/logs
log() { echo "$(date '+%F %T') $*" >> $O/progress.log; }
order() { echo $TEXT; for f in $P4/wraps/w*.sfc; do echo "$(stat -c %s $f) $(basename $f .sfc)"; done | sort -rn | awk '{print $2}' | grep -vxF "$(echo $TEXT | tr ' ' '\n')"; }
facesdone() { cat $O/faces*.tsv 2>/dev/null | grep -q "^$1 "; }
producer() {
  [ -n "$WAITPID" ] && while kill -0 $WAITPID 2>/dev/null; do sleep 20; done
  for w in $(order); do
    [ -e $O/STOP ] && break
    facesdone $w && continue
    while [ ! -e $P4/flat/$w/flatten.txt ] && [ ! -e $P4/img/${w}_full_side.jpg ]; do [ -e $O/STOP ] && return; sleep 60; done   # p4full's flat grid finished
    nice -n 5 $U faces $P4/flat $O/out --wraps $w --ct $R $K --ct-um 9.6 --umb $UMB --umb-um 2.4 --gt $GT 9.6 --gt-um 9.6 --threads $FTH \
      2>> $O/logs/faces_b.log | grep "^$w " >> $O/faces_b.tsv || log "$w faces failed"
  done
  touch $O/producer.done
}
memok() { [ $(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo) -ge $MINMEM ]; }
one() {
  local w=$1 d=$O/r/$1
  mkdir -p $d/in
  nice -n 5 $U tifxyz-scale $O/out $O/up 1 --wraps ${w}_recto,${w}_verso --flip-rows 1 > /dev/null 2>&1 || { log "$w upscale failed"; return 1; }
  ln -sfn $O/up/${w}_recto $d/in/${w}_recto; ln -sfn $O/up/${w}_verso $d/in/${w}_verso
  UFSM_RENDER_MEM=16 nice -n 5 $U render $d/in $d --volume $I 9.6 --ct $R $K --depth 0 --scale 2 --threads $TH > $d/render.log 2>&1 || { log "$w render failed"; return 1; }
  C="--ct $d/${w}_recto_ct.tif --ink recto=$d/${w}_recto.tif --ink verso=$d/${w}_verso.tif --px-um 4.8 --stretch --stretch-ct --threads $TH"
  nice -n 5 $U ink-compare $O/prev/${w}_rv_rgb.jpg $C --rgb-agree recto,verso --max-px 8000 > $d/stats.txt 2>&1
  nice -n 5 $U ink-compare $O/img/${w}_rv_rgb.jpg $C --rgb-agree recto,verso --max-px 65000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/prev/${w}_rv_side.jpg $C --max-px 8000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/img/${w}_rv_side.jpg $C --max-px 65000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/prev/${w}_rv_viridis.jpg $C --cmap viridis --no-ct --max-px 8000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/img/${w}_v.jpg $C --cmap viridis --each $O/prev/${w}_ --each-suffix _viridis --max-px 8000 > /dev/null 2>&1
  nice -n 5 $U ink-compare $O/img/${w}_v.jpg $C --cmap viridis --each $O/img/${w}_ --each-suffix _viridis --max-px 65000 > /dev/null 2>&1
  [ -e $O/img/${w}_rv_side.jpg ] && [ -e $O/img/${w}_verso_viridis.jpg ] && rm -f $d/*.tif
  log "$w rendered: $(grep -E '^corr' $d/stats.txt)"
}
producer &
consume() {   # pass 1: wraps never rendered; pass 2: wraps rendered before the viridis outputs existed
  for w in $(order); do
    [ -e $O/STOP ] && return
    [ -e $O/img/${w}_verso_viridis.jpg ] && continue
    [ $1 = 1 ] && [ -e $O/img/${w}_rv_side.jpg ] && continue
    until facesdone $w; do { [ -e $O/STOP ] || [ -e $O/producer.done ]; } && break; sleep 20; done
    facesdone $w || continue
    while [ $(jobs -r | wc -l) -ge $((SPAR + 1)) ] || ! memok; do sleep 15; done   # (+1: the producer)
    one $w &
    sleep 30
  done
}
consume 1; wait; consume 2; wait
log "all done: $(ls $O/img/*_verso_viridis.jpg 2>/dev/null | wc -l) wraps"

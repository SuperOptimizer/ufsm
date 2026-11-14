#!/bin/bash
# strip_refill.sh OUT: re-render every stripe of a finished strip (OUT/strip/flat/stripes) with hole filling and mosaic them into
# OUT/img_fill (per stripe <s>_ct.jpg + <s>_ink.jpg, grey), bounded disk like strip_stream.sh (strip-mosaic --wait-renders --rm-renders, at most MAXR render pairs on disk).
# Renders go to OUT/render_fill (removed by the mosaic after use), the per-pixel fill flags (<s>_fill.tif: 0 on the surface, k / 50+k
# filled from +-k voxels along the normal (+100 in an interpolated grid hole), 250 inpainted, 254 no surface, 255 still empty) are kept in OUT/fillmap with the render
# logs' fill shares in OUT/logs/refill_<s>.log. The contrast stretch is OUT/stretch.txt (the original run's, so both mosaics compare)
# or STRETCH. Resumable: rendered stripes still on disk are reused after a stop; stripes the stopped mosaic already consumed (and
# deleted) are rendered again, since the restarted mosaic needs every stripe.
# PAPMOS=DIR: a second mosaic in the papyrus look (strip-mosaic --papyrus PGAIN --papyrus-bg 128 --overview-px PAPPX) beside img_fill;
# made by the same strip-mosaic process from the same reads (--papyrus-dir; PGAIN default 1, PAPPX 8000).
# PAP=DIR: also a papyrus-look preview per stripe (ink-compare --papyrus, --max-px 8000) DIR/<s>.jpg, made before the mosaic takes the
# render. MEMGB: each render waits until MemAvailable exceeds it (default 80: a stripe render is ~40 GB RSS, keep 40 GB for the rest).
#   env: U, CTR CTK (CT root / key at 9.6 um), INK INKK, INKGATE (two roots, key INKK: the ink gated by their max; the latent ink head), RENDER_NT, RJOBS (concurrent renders, 1), MOS_NT, MAXR, CTL, FILLX (render fill flags), STRETCH, PAP, MEMGB;
#   STOP: OUT/STOP_FILL
set -u
OUT=$1
U=${U:-$(cd "$(dirname "$0")/../.." && pwd)/build/ufsm}   # (the repo's build; the old /vesuvius/ufsm/scroll1 default is gone)
[ -x "$U" ] || { echo "$0: no ufsm binary $U (set U)" >&2; exit 1; }
CTR=${CTR:?CT root}; CTK=${CTK:?CT key}; INK=${INK:?ink zarr}; INKK=${INKK:-9.6}
RENDER_NT=${RENDER_NT:-8}; MOS_NT=${MOS_NT:-6}; MAXR=${MAXR:-6}; [ $MAXR -ge 1 ] || MAXR=1; CTL=${CTL:-80}; PAP=${PAP:-}; MEMGB=${MEMGB:-80}; PAPMOS=${PAPMOS:-}; PGAIN=${PGAIN:-1}; PAPPX=${PAPPX:-8000}
FILLX=${FILLX:---fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 0}
R=$OUT/render_fill; I=$OUT/img_fill; M=$OUT/fillmap
mkdir -p $OUT/logs $R $M
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/progress_fill.log; }
T0=$(date +%s)
[ -e $OUT/strip/flat/flat.json ] || { log "no flattened strip"; exit 1; }
S=($(ls $OUT/strip/flat/stripes))
# INKGATE="ROOT1 ROOT2" (key INKK; the latent ink head's c12 / c13 = p_recto / p_verso, papyrus only, never supervised in air): every
# stripe's ink render x max(p1, p2) / 255 (air -> 0; inkB's GATE=1), marked <s>.gated; only when both stores exist
GATED=0; if [ -n "${INKGATE:-}" ]; then GATED=1; for gr in $INKGATE; do [ -e $gr/zarr.json ] || GATED=0; done; fi
gate() {   # gate S: the ink render R/S.tif (its stripe linked in R/in_S) x max(gates)
  local s=$1 gi=0 gr; [ $GATED = 1 ] || return 0; [ -e $R/$s.gated ] && return 0
  local own=0; [ -d $R/in_$s ] || { own=1; mkdir -p $R/in_$s; ln -sfn $OUT/strip/flat/stripes/$s $R/in_$s/$s; }
  for gr in $INKGATE; do gi=$((gi + 1)); mkdir -p $R/gate$gi
    UFSM_RENDER_MEM=16 nice -n 10 $U render $R/in_$s $R/gate$gi --volume $gr $INKK --ct $CTR $CTK --depth 1 --scale 1 --threads $RENDER_NT $FILLX > $OUT/logs/refill_${s}_gate$gi.log 2>&1 || { log "gate render $s $gi FAILED"; return 1; }
    rm -f $R/gate$gi/${s}_ct.tif $R/gate$gi/${s}_fill.tif; done
  python3 -I -c "import sys, os, numpy as np, tifffile as t; r, s = sys.argv[1:3]; i = t.imread(f'{r}/{s}.tif'); p = np.maximum(t.imread(f'{r}/gate1/{s}.tif'), t.imread(f'{r}/gate2/{s}.tif')).astype(np.float32) / 255
t.imwrite(f'{r}/{s}.tif.g', (i.astype(np.float32) * p + 0.5).astype(i.dtype), compression=None); os.replace(f'{r}/{s}.tif.g', f'{r}/{s}.tif')" $R $s || { log "gate $s FAILED"; return 1; }
  rm -f $R/gate1/$s.tif $R/gate2/$s.tif; [ $own = 1 ] && rm -r $R/in_$s; touch $R/$s.gated
}
if [ $GATED = 1 ] && [ -z "${STRETCH:-}" ] && [ ! -e $OUT/stretch.gated ]; then
  # the contrast stretch from the GATED renders of the 3 middle stripes (the driver's stretch.txt came from the ungated ones; gated air at 0
  # would pull it): CT 0.5 / 99.8 % over rendered pixels (CT > 0), ink 1 / 99.9 % over papyrus pixels only (CT >= CTL), lead 11-13
  N=${#S[@]}; mids=(); for k in $((N / 2 - 1)) $((N / 2)) $((N / 2 + 1)); do [ $k -ge 0 ] && [ $k -lt $N ] && mids+=(${S[$k]}); done
  for s in "${mids[@]}"; do [ -e $R/$s.done ] || { mkdir -p $R/in_$s; ln -sfn $OUT/strip/flat/stripes/$s $R/in_$s/$s
      UFSM_RENDER_MEM=16 nice -n 10 $U render $R/in_$s $R --volume $INK $INKK --ct $CTR $CTK --depth 1 --scale 1 --threads $RENDER_NT $FILLX > $OUT/logs/refill_$s.log 2>&1 || { log "render $s FAILED"; exit 1; }
      [ -e $R/${s}_fill.tif ] && mv $R/${s}_fill.tif $M/; rm -r $R/in_$s; touch $R/$s.done; }
    gate $s || exit 1; done
  st=$(python3 -I -c "import sys, numpy as np, tifffile as t; r, ctl = sys.argv[1], int(sys.argv[2]); c = []; k = []
for s in sys.argv[3:]:
    a = t.imread(f'{r}/{s}_ct.tif'); i = t.imread(f'{r}/{s}.tif'); c.append(a[a > 0][::7]); k.append(i[a >= ctl][::7])
c = np.concatenate(c); k = np.concatenate(k); q = lambda v, lo, hi: (int(np.percentile(v, lo)), max(int(np.percentile(v, hi)), int(np.percentile(v, lo)) + 1))
ca, cb = q(c, 0.5, 99.8); ia, ib = q(k, 1, 99.9); print(f'{ca},{cb},{ia},{ib}')" $R $CTL "${mids[@]}") || { log "gated stretch FAILED"; exit 1; }
  log "stretch from the gated middle stripes (papyrus-only ink): $st (was $(cat $OUT/stretch.txt 2> /dev/null))"
  cp -f $OUT/stretch.txt $OUT/stretch.ungated.txt 2> /dev/null; echo $st > $OUT/stretch.txt; touch $OUT/stretch.gated
fi
for d in $R/*.done; do [ -e "$d" ] || continue; s=$(basename $d .done); [ -e $R/${s}_ct.tif ] && { gate $s || exit 1; }; done   # (renders reused from the driver)
ST=${STRETCH:-$(cat $OUT/stretch.txt 2> /dev/null)}; [ -n "$ST" ] || { log "no stretch (OUT/stretch.txt or STRETCH)"; exit 1; }
rend() {   # one stripe -> render_fill/<s>.tif, <s>_ct.tif, <s>.done; fillmap/<s>_fill.tif
  local s=$1; [ -e $R/$s.done ] && return 0
  while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt $MEMGB ]; do sleep 30; done
  mkdir -p $R/in_$s; ln -sfn $OUT/strip/flat/stripes/$s $R/in_$s/$s
  UFSM_RENDER_MEM=16 nice -n 10 $U render $R/in_$s $R --volume $INK $INKK --ct $CTR $CTK --depth 1 --scale 1 --threads $RENDER_NT $FILLX > $OUT/logs/refill_$s.log 2>&1 || { log "render $s FAILED"; return 1; }
  [ -e $R/${s}_fill.tif ] && mv $R/${s}_fill.tif $M/
  gate $s || return 1
  if [ -n "$PAP" ]; then mkdir -p $PAP
    nice -n 10 $U ink-compare $PAP/$s.jpg --ct $R/${s}_ct.tif --ink ink=$R/$s.tif --papyrus ink --papyrus-gain 1 --papyrus-bg 128 --stretch --stretch-ct \
      --px-um 9.6 --max-px 8000 --threads $RENDER_NT > $OUT/logs/refill_pap_$s.log 2>&1 || log "preview $s failed (exit $?)"; fi
  rm -r $R/in_$s; touch $R/$s.done; [ $GATED = 1 ] && touch $R/$s.gated; log "render $s done: $(grep -o 'void on the surface.*' $OUT/logs/refill_$s.log | head -1)"
}
for d in $R/*.done; do [ -e "$d" ] || continue; s=$(basename $d .done); [ -e $R/${s}_ct.tif ] || rm -f $d; done   # the mosaic needs every stripe: re-render those it already consumed
# one mosaic process; PAPMOS: the papyrus mosaic from the same reads (--papyrus-dir: the files of a separate --papyrus run)
PAPX=(); [ -n "$PAPMOS" ] && PAPX=(--papyrus-dir $PAPMOS --papyrus-gain $PGAIN --papyrus-bg 128 --papyrus-overview-px $PAPPX)
nice -n 10 $U strip-mosaic $OUT/strip/flat $R $I --ct-level $CTL --threads $MOS_NT --stretch $ST --wait-renders 1 --rm-renders 1 --fill-maps $M "${PAPX[@]}" > $OUT/logs/mosaic_fill.log 2>&1 &
MPS=($!)
clean() { :; }   # (the mosaic removes each render after use)
alive() { local p; for p in "${MPS[@]}"; do kill -0 $p 2> /dev/null || return 1; done; return 0; }
# RJOBS renders at once (default 1; each RENDER_NT threads): a stripe render uses ~13 of 20 threads and ~15% of the GPU (0490B), so two
# overlap; the mosaic takes stripes in order by their .done markers, whichever render finishes first
RP=(); RFAIL=0
reap() { local q=() p; for p in "${RP[@]}"; do if kill -0 $p 2> /dev/null; then q+=($p); else wait $p || RFAIL=1; fi; done; RP=("${q[@]}"); }
for s in "${S[@]}"; do
  [ -e $OUT/STOP_FILL ] && { kill "${MPS[@]}"; wait; exit 0; }
  clean; reap
  # only renders the mosaic takes before s count against MAXR: renders already on disk past s (the stretch's 3 middle stripes) wait for
  # the mosaic, which waits for s, so counting them deadlocked whenever MAXR <= their number
  while { [ $(( $(ls $R | grep '_ct\.tif$' | awk -v s=$s '$0 < s' | wc -l) + ${#RP[@]} )) -ge $MAXR ] || [ ${#RP[@]} -ge ${RJOBS:-1} ]; } && alive && [ $RFAIL = 0 ]; do sleep 2; clean; reap; done
  [ $RFAIL = 0 ] || { kill "${MPS[@]}"; wait; exit 1; }
  alive || break
  if [ ${RJOBS:-1} -gt 1 ]; then rend $s & RP+=($!); else rend $s || { kill "${MPS[@]}"; exit 1; }; fi
done
for p in "${RP[@]}"; do wait $p || RFAIL=1; done
[ $RFAIL = 0 ] || { kill "${MPS[@]}"; exit 1; }
for p in "${MPS[@]}"; do while kill -0 $p 2> /dev/null; do sleep 10; clean; done; wait $p || { log "mosaic FAILED"; exit 1; }; done
clean
log "mosaic done: $(grep "^strip-mosaic: .* overview $I/" $OUT/logs/mosaic_fill.log | tail -1)"
[ -n "$PAPMOS" ] && log "papyrus mosaic done: $(grep "^strip-mosaic: .* overview $PAPMOS/" $OUT/logs/mosaic_fill.log | tail -1)"
log "refill render + mosaic in $(( $(date +%s) - T0 )) s"

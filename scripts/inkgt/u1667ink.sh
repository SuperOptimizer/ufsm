#!/bin/bash
# scripts/inkgt/u1667ink.sh CKPT TAG: a latent checkpoint's ink head over a u1 unroll's centre surface (SCROLL, default PHerc1667),
# scored against the published human ink labels like ift3-24000 (inkscore-u1.tsv: 40 um, ink AUC 0.646). inkB, 2026-11-12.
#   1. predict --heads ink (the latent multi-head path, as scripts/univ/eval.sh INKP=heads) at level LV (2 = 9.596 um, the unroll's grid)
#      over the box of the labelled segments (+ 64 voxels; a scroll without labels: the stripes' box), air shards skipped (--occupancy 5,
#      as unroll.sh's ink) -> SCROLL-u1/ink/TAG.h/ink
#   2. render every centre stripe (strip/strip/flat/stripes) through it, depth 1 + the strip's fill (FILLX, as strip/img_fill)
#      -> SCROLL-u1/inkr/TAG/<s>.tif, <s>_ct.tif (fill maps in inkr/TAG/fillmap)
#   3. labels (inkgt/segs-SCROLL.json): scripts/inkgt/wrapscore.py on those renders -> sched/inkscore-u1-TAG.tsv (inkscore.sh columns)
#   4. strip-mosaic: stripe JPEGs + overview for viewing -> SCROLL-u1/inkr/TAG/img (the contrast stretch from these renders; renders kept)
# GPU 0 only: steps 1-2 wait for flock /tmp/gpu0-bench.lock (the desk's GPU-0 queue), so they run behind other GPU-0 jobs.
# Resumable: a finished predict (ink.done) is kept (predict --heads takes no --resume), rendered stripes are kept, the score reruns. Env: SCROLL, U (ufsm), LV (2), MEMR (render memcap, 48G), TH (12).
set -u
CK=$(readlink -f "${1:?ckpt}"); TAG=${2:?tag}; SCROLL=${SCROLL:-PHerc1667}
case $SCROLL in
  PHerc1667) K=PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr; UMV=2.399 ;;
  PHerc1299) K=PHerc1299/20260309130042-2.399um-0.2m-78keV-masked.zarr; UMV=2.399 ;;
  PHercParis4) K=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; UMV=2.400 ;;
  *) echo "u1667ink: no CT key for $SCROLL (add it to the case)"; exit 1 ;;
esac
O=/vesuvius/ufsm/unroll/$SCROLL-u1; SD=/vesuvius/ufsm/unroll/sched; U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}; LV=${LV:-2}; TH=${TH:-12}
R=/vesuvius/ufsm/v2/ct; PY=$(dirname $(readlink -f $0))/wrapscore.py; SEGS=/vesuvius/ufsm/inkgt/segs-$SCROLL.json; [ -s $SEGS ] || SEGS=
# STRIPES="ws020 ws021": only these stripes (box from their bboxes, scored on them only, per-stripe JPEGs instead of the strip-mosaic)
STRIPES=${STRIPES:-}; SDIR=$O/strip/strip/flat/stripes
FILLX=${FILLX:---fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 0}
INK=$O/ink/$TAG.h; RD=$O/inkr/$TAG; mkdir -p $O/ink $RD/logs $RD/fillmap $RD/img
if [ -n "$STRIPES" ]; then mkdir -p $RD/stripes; for s in $STRIPES; do [ -d $SDIR/$s ] || { echo "u1667ink: no stripe $s"; exit 1; }; ln -sfn $SDIR/$s $RD/stripes/$s; done; SDIR=$RD/stripes; fi
ST=$SD/inkscore-u1-$TAG; [ $SCROLL = PHerc1667 ] || ST=$SD/inkscore-$SCROLL-u1-$TAG
key=$(python3 -I -c "print(round($UMV * 2 ** $LV, 3))")
# before each GPU step: wait for the WAITF files (space-separated, e.g. another job's .done) and for no other GPU-0 process above 200 MiB
WAITF=${WAITF:-}
gpu0_wait() { local f u n=0; for f in $WAITF; do while [ ! -e $f ]; do [ $n = 0 ] && log "waiting for $f"; n=1; sleep 60; done; done
  u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
  while nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u=$u '$1 == u && $2 > 200 { b = 1 } END { exit !b }'; do
    [ $n -lt 2 ] && log "waiting for GPU 0 (a process above 200 MiB)"; n=2; sleep 60; done; }
log() { echo "$(date '+%F %T') u1667ink $SCROLL $TAG: $*" | tee -a $RD/logs/run.log; }
[ -s $CK ] || { log "no checkpoint $CK"; exit 1; }
# the box at level LV, zyx, + 64 voxels, clipped to the unroll box (level-2 voxels): the labelled segments' tifxyz bboxes (level-0 xyz),
# or without labels the stripes' bboxes (xyz in voxel_size_um voxels)
box=$(python3 -I - $LV $O/box.txt "$([ -n "$STRIPES" ] || echo $SEGS)" $SDIR $UMV <<'PY'
import json, sys, os, glob
lv, bx, segs, sd, um = int(sys.argv[1]), [int(v) for v in open(sys.argv[2]).read().split(",")], sys.argv[3], sys.argv[4], float(sys.argv[5])
lo = [1e30] * 3; hi = [-1e30] * 3; f = 2 ** lv
if segs: metas = [(os.path.join(g["tifxyz"], "meta.json"), 1.0) for g in json.load(open(segs))["segments"]]   # level-0 voxels
else: metas = [(m, None) for m in glob.glob(os.path.join(sd, "*", "meta.json"))]
for p, v in metas:
    m = json.load(open(p)); b = m["bbox"]   # [[x, y, z] min, max]
    sc = v if v is not None else m["voxel_size_um"] / um   # -> level-0 voxels
    for d, k in enumerate((2, 1, 0)): lo[d] = min(lo[d], b[0][k] * sc); hi[d] = max(hi[d], b[1][k] * sc)
s = 4 / f   # (the unroll box is in level-2 voxels)
a = [max(int(bx[d] * s), int(lo[d] / f) - 64) for d in range(3)]; e = [min(int((bx[d] + bx[3 + d]) * s), int(hi[d] / f) + 64) for d in range(3)]
print(",".join(map(str, a + [e[d] - a[d] for d in range(3)])))
PY
) || { log "box failed"; exit 1; }
log "ckpt $CK, level $LV ($key um), box $box, labels ${SEGS:-none (render only)}"
if [ ! -e $INK/ink.done ]; then
  # BLEND=1 (default): the latent ink head's level depends on its window (inkpod-9000: steps of 10-30 / 255 at the window seams, halo 16..256
  # and --gn-frozen alike), so two passes on tile grids offset by half a tile (--grid-origin 256) are blended with sin^2 weights (ink-blend:
  # seam step 16.9 -> 1.3 on a 1299 test box); BLEND=0 one pass
  rm -rf $INK; mkdir -p $INK
  # GATE=1: also the recto / verso probability channels (c12, c13: papyrus only, never supervised in air) for a papyrus-gated ink render
  HEADS=ink; HL="ink"; [ "${GATE:-0}" = 1 ] && { HEADS=ink,c12,c13; HL="ink c12 c13"; }
  # (--shard 256 on the shifted pass: its tiles align with the store shards; 1299 test box byte-identical, 63 s -> 5 s)
  # pbox SHIFT: the box moved down onto the pass's 512 tile grid (origin = SHIFT mod 512), as seamless.c pass_box (an off-grid box: one
  # window per shard, ~6-8x the windows); --axis: the radial input channels (predict refuses without it since 2026-11-14); --q 0 --levels 1:
  # the passes only feed ink-blend (level 0)
  pbox() { awk -F, -v s=$1 '{ for (d = 1; d <= 3; d++) { l = $d - s; l = (l >= 0 ? int(l / 512) : -int((-l + 511) / 512)) * 512 + s; if (l < 0) l = s
           o[d] = l; e = $d + $(d + 3); n[d] = e > l ? e - l : 1 } printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], n[1], n[2], n[3] }' <<< "$box"; }
  AX=${AXIS:-$O/axis.json}; [ -s $AX ] || { log "no axis $AX"; exit 1; }
  PQ="--q 0 --levels 1"; [ "${BLEND:-1}" = 1 ] || PQ=
  for g in 0 $([ "${BLEND:-1}" = 1 ] && echo 256); do
    log "predict grid $g (waiting for GPU 0)"; gpu0_wait
    flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 ~/bin/memcap -m 40G nice -n 10 $U predict $CK $R $K $INK/g$g --um $UMV --level $LV --box $(pbox $g) \
      --full-grid 1 --occupancy 5 --heads $HEADS --gpu 0 --threads $TH --window 544 --halo 16 --grid-origin $g,$g,$g $([ $g = 0 ] || echo --shard 256) --axis $AX $PQ > $RD/logs/predict_g$g.log 2>&1 ||
      { log "predict grid $g failed ($RD/logs/predict_g$g.log)"; exit 1; }
    [ -e $INK/g$g/ink/zarr.json ] || { log "no ink head output in $INK/g$g"; exit 1; }
  done
  if [ "${BLEND:-1}" = 1 ]; then
    log "ink-blend"
    for h in $HL; do
      ~/bin/lowprio ~/bin/memcap -m 24G $U ink-blend $INK/$h --um $key --tile 512,512,512 --grids 0,0,0:256,256,256 $INK/g0/$h $INK/g256/$h --threads 8 \
        >> $RD/logs/blend.log 2>&1 || { log "ink-blend $h failed ($RD/logs/blend.log)"; exit 1; }
      [ -e $INK/$h/zarr.json ] || { log "no blended $h in $INK/$h"; exit 1; }
    done; rm -rf $INK/g0 $INK/g256
  else for h in $HL; do mv $INK/g0/$h $INK/$h; done; rm -rf $INK/g0; fi
  touch $INK/ink.done; log "predict done"
fi
for st in $SDIR/*/; do s=$(basename $st); [ -e $RD/$s.done ] && continue
  rm -rf $RD/in_$s; mkdir -p $RD/in_$s; ln -sfn $st $RD/in_$s/$s
  log "render $s (waiting for GPU 0)"; gpu0_wait
  flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 UFSM_RENDER_MEM=16 UFSM_RENDER_GPU_MEM=4 ~/bin/memcap -m ${MEMR:-48G} nice -n 10 \
    $U render $RD/in_$s $RD --volume $INK/ink $key --ct $R $K/$LV --depth 1 --scale 1 --threads $TH $FILLX > $RD/logs/render_$s.log 2>&1 ||
    { log "render $s failed ($RD/logs/render_$s.log)"; exit 1; }
  [ -e $RD/${s}_fill.tif ] && mv $RD/${s}_fill.tif $RD/fillmap/
  if [ "${GATE:-0}" = 1 ]; then for h in c12 c13; do mkdir -p $RD/$h
    flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 UFSM_RENDER_MEM=16 UFSM_RENDER_GPU_MEM=4 ~/bin/memcap -m ${MEMR:-48G} nice -n 10 \
      $U render $RD/in_$s $RD/$h --volume $INK/$h $key --ct $R $K/$LV --depth 1 --scale 1 --threads $TH $FILLX > $RD/logs/render_${s}_$h.log 2>&1 ||
      { log "render $s $h failed ($RD/logs/render_${s}_$h.log)"; exit 1; }
    rm -f $RD/$h/${s}_ct.tif $RD/$h/${s}_fill.tif; done; fi
  rm -rf $RD/in_$s; touch $RD/$s.done
done
if [ -n "$SEGS" ]; then
  log "score"
  CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m 24G python3 -I $PY $SDIR $SEGS $ST.tsv \
    --map $RD --ct-key $K --step 8 --tol 20,40,80 --gate-auc ${GATE_AUC:-0} > $ST.log 2>&1; rc=$?
  [ $rc = 0 ] || [ $rc = 3 ] || { log "score failed ($ST.log)"; exit 1; }
  log "scored: $ST.tsv: $(grep '^HEADLINE' $ST.log | cut -c10-)"; [ $rc = 3 ] && { log "GATE: $(grep '^GATE' $ST.log)"; exit 3; }
fi
if [ -n "$STRIPES" ]; then   # (strip-mosaic needs every stripe) per stripe: CT and ink JPEGs at full res (1 / 99.5 % stretch) + a 1:4 CT | ink overview
  log "jpegs"
  ~/bin/lowprio python3 -I - $RD $STRIPES <<'PY' >> $RD/logs/run.log 2>&1 || { log "jpegs failed"; exit 1; }
import os, sys, numpy as np, tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None; rd = sys.argv[1]
def st(a):
    v = a[a > 0]; lo, hi = (np.percentile(v, 1), np.percentile(v, 99.5)) if v.size else (0, 255)
    return (np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
ov = []
for s in sys.argv[2:]:
    c, i = st(tifffile.imread(f"{rd}/{s}_ct.tif")), st(tifffile.imread(f"{rd}/{s}.tif"))
    Image.fromarray(c).save(f"{rd}/img/{s}_ct.jpg", quality=88); Image.fromarray(i).save(f"{rd}/img/{s}_ink.jpg", quality=88); row = [c, i]
    if os.path.exists(f"{rd}/c12/{s}.tif"):   # papyrus-gated: ink x max(p_recto, p_verso) (air -> 0; the head is never supervised in air)
        pp = np.maximum(tifffile.imread(f"{rd}/c12/{s}.tif"), tifffile.imread(f"{rd}/c13/{s}.tif")).astype(np.float32) / 255
        g = st((tifffile.imread(f"{rd}/{s}.tif").astype(np.float32) * pp).astype(np.uint8)); Image.fromarray(g).save(f"{rd}/img/{s}_inkgated.jpg", quality=88); row.append(g)
    h, w = c.shape; ov.append(np.concatenate([np.asarray(Image.fromarray(x).resize((w // 4, h // 4), Image.BILINEAR)) for x in row], 0))
W = max(o.shape[1] for o in ov); ov = np.concatenate([np.pad(o, ((0, 8), (0, W - o.shape[1]))) for o in ov], 0)
Image.fromarray(ov).save(f"{rd}/img/overview.jpg", quality=88); print("overview", ov.shape)
PY
  log "done: $RD/img/overview.jpg"
else
log "mosaic"
CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m 24G $U strip-mosaic $O/strip/strip/flat $RD $RD/img --ct-level 80 --threads 6 --fill-maps $RD/fillmap \
  > $RD/logs/mosaic.log 2>&1 || { log "mosaic failed ($RD/logs/mosaic.log)"; exit 1; }
log "done: $RD/img/overview.jpg $(tail -1 $RD/logs/mosaic.log)"
fi
[ -n "$SEGS" ] || exit 0
# held out only (validation + holdout boxes); the training labels' AUC ("rest") beside it, never "all" (~76 % training samples, audit 11-14 #4)
printf 'set\ttol\tcoverage(heldout)\tink_auc(heldout)\tink_f1best(heldout)\tct_auc(heldout)\tink_auc(rest=train)\n'
for t in $SD/inkscore-u1 $ST; do [ -s $t.tsv ] || continue
  awk -F'\t' -v t=$(basename $t) '$2 == "rest" { r[$1] = $7 } $2 == "validation" || $2 == "holdout" || $2 == "heldout" { h[$1, $2] = $0 }
    END { for (k in h) { split(k, q, SUBSEP); if (q[2] != "heldout") continue; split(h[k], f, "\t")
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", (t == "inkscore-u1" ? "ift3-24000" : t), f[1], f[5], f[7], f[8], f[9], r[f[1]] } }' $t.tsv; done | sort -k1,1 -k2,2n

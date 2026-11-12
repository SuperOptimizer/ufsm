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
# Resumable: the predict resumes, rendered stripes are kept, the score reruns. Env: SCROLL, U (ufsm), LV (2), MEMR (render memcap, 48G), TH (12).
set -u
CK=$(readlink -f "${1:?ckpt}"); TAG=${2:?tag}; SCROLL=${SCROLL:-PHerc1667}
case $SCROLL in
  PHerc1667) K=PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr ;;
  PHerc1299) K=PHerc1299/20260309130042-2.399um-0.2m-78keV-masked.zarr ;;
  *) echo "u1667ink: no CT key for $SCROLL (add it to the case)"; exit 1 ;;
esac
O=/vesuvius/ufsm/unroll/$SCROLL-u1; SD=/vesuvius/ufsm/unroll/sched; U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}; LV=${LV:-2}; TH=${TH:-12}
R=/vesuvius/ufsm/v2/ct; PY=$(dirname $(readlink -f $0))/wrapscore.py; SEGS=/vesuvius/ufsm/inkgt/segs-$SCROLL.json; [ -s $SEGS ] || SEGS=
FILLX=${FILLX:---fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 1}
INK=$O/ink/$TAG.h; RD=$O/inkr/$TAG; mkdir -p $O/ink $RD/logs $RD/fillmap $RD/img
ST=$SD/inkscore-u1-$TAG; [ $SCROLL = PHerc1667 ] || ST=$SD/inkscore-$SCROLL-u1-$TAG
key=$(python3 -I -c "print(round(2.399 * 2 ** $LV, 3))")
log() { echo "$(date '+%F %T') u1667ink $SCROLL $TAG: $*" | tee -a $RD/logs/run.log; }
[ -s $CK ] || { log "no checkpoint $CK"; exit 1; }
# the box at level LV, zyx, + 64 voxels, clipped to the unroll box (level-2 voxels): the labelled segments' tifxyz bboxes (level-0 xyz),
# or without labels the stripes' bboxes (xyz in voxel_size_um voxels)
box=$(python3 -I - $LV $O/box.txt "$SEGS" $O/strip/strip/flat/stripes <<'PY'
import json, sys, os, glob
lv, bx, segs, sd = int(sys.argv[1]), [int(v) for v in open(sys.argv[2]).read().split(",")], sys.argv[3], sys.argv[4]
lo = [1e30] * 3; hi = [-1e30] * 3; f = 2 ** lv
if segs: metas = [(os.path.join(g["tifxyz"], "meta.json"), 1.0) for g in json.load(open(segs))["segments"]]   # level-0 voxels
else: metas = [(m, None) for m in glob.glob(os.path.join(sd, "*", "meta.json"))]
for p, v in metas:
    m = json.load(open(p)); b = m["bbox"]   # [[x, y, z] min, max]
    sc = v if v is not None else m["voxel_size_um"] / 2.399   # -> level-0 voxels
    for d, k in enumerate((2, 1, 0)): lo[d] = min(lo[d], b[0][k] * sc); hi[d] = max(hi[d], b[1][k] * sc)
s = 4 / f   # (the unroll box is in level-2 voxels)
a = [max(int(bx[d] * s), int(lo[d] / f) - 64) for d in range(3)]; e = [min(int((bx[d] + bx[3 + d]) * s), int(hi[d] / f) + 64) for d in range(3)]
print(",".join(map(str, a + [e[d] - a[d] for d in range(3)])))
PY
) || { log "box failed"; exit 1; }
log "ckpt $CK, level $LV ($key um), box $box, labels ${SEGS:-none (render only)}"
if [ ! -e $INK/ink.done ]; then
  log "predict (waiting for GPU 0)"
  flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 ~/bin/memcap -m 40G nice -n 10 $U predict $CK $R $K $INK --um 2.399 --level $LV --box $box \
    --full-grid 1 --resume 1 --occupancy 5 --heads ink --gpu 0 --threads $TH > $RD/logs/predict.log 2>&1 || { log "predict failed ($RD/logs/predict.log)"; exit 1; }
  [ -e $INK/ink/zarr.json ] || { log "no ink head output in $INK (logs/predict.log)"; exit 1; }
  touch $INK/ink.done; log "predict done"
fi
for st in $O/strip/strip/flat/stripes/*/; do s=$(basename $st); [ -e $RD/$s.done ] && continue
  rm -rf $RD/in_$s; mkdir -p $RD/in_$s; ln -sfn $st $RD/in_$s/$s
  log "render $s (waiting for GPU 0)"
  flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 UFSM_RENDER_MEM=16 UFSM_RENDER_GPU_MEM=4 ~/bin/memcap -m ${MEMR:-48G} nice -n 10 \
    $U render $RD/in_$s $RD --volume $INK/ink $key --ct $R $K/$LV --depth 1 --scale 1 --threads $TH $FILLX > $RD/logs/render_$s.log 2>&1 ||
    { log "render $s failed ($RD/logs/render_$s.log)"; exit 1; }
  rm -rf $RD/in_$s; [ -e $RD/${s}_fill.tif ] && mv $RD/${s}_fill.tif $RD/fillmap/; touch $RD/$s.done
done
if [ -n "$SEGS" ]; then
  log "score"
  CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m 24G python3 -I $PY $O/strip/strip/flat/stripes $SEGS $ST.tsv \
    --map $RD --ct-key $K --step 8 --tol 20,40,80 > $ST.log 2>&1 || { log "score failed ($ST.log)"; exit 1; }
  log "scored: $ST.tsv"
fi
log "mosaic"
CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m 24G $U strip-mosaic $O/strip/strip/flat $RD $RD/img --ct-level 80 --threads 6 --fill-maps $RD/fillmap \
  > $RD/logs/mosaic.log 2>&1 || { log "mosaic failed ($RD/logs/mosaic.log)"; exit 1; }
log "done: $RD/img/overview.jpg $(tail -1 $RD/logs/mosaic.log)"
[ -n "$SEGS" ] || exit 0
printf 'set\ttol\tcoverage\tink_auc\tink_f1best\tct_auc\n'
for t in $SD/inkscore-u1 $ST; do awk -F'\t' -v t=$(basename $t) '$2 == "all" { printf "%s\t%s\t%s\t%s\t%s\t%s\n", (t == "inkscore-u1" ? "ift3-24000" : t), $1, $5, $7, $8, $9 }' $t.tsv; done

#!/bin/bash
# scripts/inkgt/u1667ink.sh CKPT TAG: a latent checkpoint's ink head over the PHerc1667 u1 unroll's centre surface, scored against the
# published human ink labels like ift3-24000 (inkscore-u1.tsv: 40 um, ink AUC 0.646). inkB, 2026-11-12.
#   1. predict --heads ink (the latent multi-head path, as scripts/univ/eval.sh INKP=heads) at level LV (2 = 9.596 um, the unroll's grid)
#      over the box of the labelled segments (+ 64 voxels) -> PHerc1667-u1/ink/TAG.h/ink
#   2. render every centre stripe (strip/strip/flat/stripes) through it, depth 1 + the strip's fill (FILLX, as strip/img_fill)
#      -> PHerc1667-u1/inkr/TAG/<s>.tif, <s>_ct.tif
#   3. scripts/inkgt/wrapscore.py on those renders -> sched/inkscore-u1-TAG.tsv (the inkscore.sh columns)
# GPU 0 only: steps 1-2 wait for flock /tmp/gpu0-bench.lock (the desk's GPU-0 queue), so they run behind other GPU-0 jobs.
# Resumable: the predict resumes, rendered stripes are kept, the score reruns. Env: U (ufsm), LV (2), MEMR (render memcap, 48G), TH (12).
set -u
CK=$(readlink -f "${1:?ckpt}"); TAG=${2:?tag}
O=/vesuvius/ufsm/unroll/PHerc1667-u1; SD=/vesuvius/ufsm/unroll/sched; U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}; LV=${LV:-2}; TH=${TH:-12}
R=/vesuvius/ufsm/v2/ct; K=PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr; PY=$(dirname $(readlink -f $0))/wrapscore.py
FILLX=${FILLX:---fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 1}
INK=$O/ink/$TAG.h; RD=$O/inkr/$TAG; mkdir -p $O/ink $RD/logs
key=$(python3 -I -c "print(round(2.399 * 2 ** $LV, 3))")
log() { echo "$(date '+%F %T') u1667ink $TAG: $*" | tee -a $RD/logs/run.log; }
[ -s $CK ] || { log "no checkpoint $CK"; exit 1; }
# the labelled segments' box (their tifxyz bboxes, level-0 xyz) at level LV, zyx, + 64 voxels, clipped to the unroll box
box=$(python3 -I - $LV $O/box.txt <<'PY'
import json, sys, os
lv, bx = int(sys.argv[1]), [int(v) for v in open(sys.argv[2]).read().split(",")]
j = json.load(open("/vesuvius/ufsm/inkgt/segs-PHerc1667.json")); lo = [1e30] * 3; hi = [-1e30] * 3
for g in j["segments"]:
    b = json.load(open(os.path.join(g["tifxyz"], "meta.json")))["bbox"]   # [[x, y, z] min, max]
    for d, k in enumerate((2, 1, 0)): lo[d] = min(lo[d], b[0][k]); hi[d] = max(hi[d], b[1][k])
f = 2 ** lv; s = 4 / f   # (the unroll box is in level-2 voxels)
a = [max(int(bx[d] * s), int(lo[d] / f) - 64) for d in range(3)]; e = [min(int((bx[d] + bx[3 + d]) * s), int(hi[d] / f) + 64) for d in range(3)]
print(",".join(map(str, a + [e[d] - a[d] for d in range(3)])))
PY
) || { log "box failed"; exit 1; }
log "ckpt $CK, level $LV ($key um), box $box"
if [ ! -e $INK/ink.done ]; then
  log "predict (waiting for GPU 0)"
  flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 ~/bin/memcap -m 40G nice -n 10 $U predict $CK $R $K $INK --um 2.399 --level $LV --box $box \
    --full-grid 1 --resume 1 --heads ink --gpu 0 --threads $TH > $RD/logs/predict.log 2>&1 || { log "predict failed ($RD/logs/predict.log)"; exit 1; }
  [ -e $INK/ink/zarr.json ] || { log "no ink head output in $INK (logs/predict.log)"; exit 1; }
  touch $INK/ink.done; log "predict done"
fi
for st in $O/strip/strip/flat/stripes/*/; do s=$(basename $st); [ -e $RD/$s.done ] && continue
  rm -rf $RD/in_$s; mkdir -p $RD/in_$s; ln -sfn $st $RD/in_$s/$s
  log "render $s (waiting for GPU 0)"
  flock /tmp/gpu0-bench.lock env CUDA_VISIBLE_DEVICES=0 UFSM_RENDER_MEM=16 UFSM_RENDER_GPU_MEM=4 ~/bin/memcap -m ${MEMR:-48G} nice -n 10 \
    $U render $RD/in_$s $RD --volume $INK/ink $key --ct $R $K/$LV --depth 1 --scale 1 --threads $TH $FILLX > $RD/logs/render_$s.log 2>&1 ||
    { log "render $s failed ($RD/logs/render_$s.log)"; exit 1; }
  rm -rf $RD/in_$s; [ -e $RD/${s}_fill.tif ] && mv $RD/${s}_fill.tif $RD/logs/; touch $RD/$s.done
done
log "score"
CUDA_VISIBLE_DEVICES= ~/bin/lowprio ~/bin/memcap -m 24G python3 -I $PY $O/strip/strip/flat/stripes /vesuvius/ufsm/inkgt/segs-PHerc1667.json $SD/inkscore-u1-$TAG.tsv \
  --map $RD --ct-key $K --step 8 --tol 20,40,80 > $SD/inkscore-u1-$TAG.log 2>&1 || { log "score failed ($SD/inkscore-u1-$TAG.log)"; exit 1; }
log "done: $SD/inkscore-u1-$TAG.tsv"
printf 'set\ttol\tcoverage\tink_auc\tink_f1best\tct_auc\n'
for t in u1 u1-$TAG; do awk -F'\t' -v t=$t '$2 == "all" { printf "%s\t%s\t%s\t%s\t%s\t%s\n", (t == "u1" ? "ift3-24000" : t), $1, $5, $7, $8, $9 }' $SD/inkscore-$t.tsv; done

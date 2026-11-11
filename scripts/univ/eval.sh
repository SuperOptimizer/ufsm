#!/usr/bin/env bash
# scripts/univ/eval.sh CKPT TAG: the universal latent net's task x resolution matrix (2026-11-11; DESIGN.md "Universal latent net").
# Columns (TASKS, default "wind dlv xres rv fib ink"; a task the checkpoint has no head for is skipped with WIND=0 / RV=0 / FIB=0 / INK=0):
#   wind  winding: wbench core (44 boxes; stats on the 39 clean) at the boxes' 9.6 um operating point (DLV 0)               -> res/TAG.tsv
#   dlv   the same boxes fed at other scales (DLVS, default "-1 1": 4.8 / 19.2 um; -2 = 2.4 um native, 8x the voxels of -1);
#         Paris 4 from the mirror for DLV != 0 (the desk's pooled CT has level 2 only)                                        -> res/TAG-dlvD.tsv
#   xres  cross-res: the 17 9.36 um GT boxes (0139 / 0814 / 0500P2) from their registered 2.2-2.4 um scans (scripts/univ/xres_prep.sh),
#         fed at 9.6 um (DLV 0) and 4.8 um (DLV -1), paired against the checkpoint's own 9.36 um rows (BOXES=all of boxes.tsv)    -> res/TAG-xres*.tsv
#   rv    recto / verso: scripts/rv/eval.sh on p1667 / p139 / p4 (RVSC)                                                          -> $RVE/TAG.txt
#   fib   fibres: predict FIBCH (H V [dyad]) on the Paris 4 HOLD box, fiber-trace (tube / --dyad), fiber-judge vs the phase GT         -> $E/TAG/fib/
#   ink   ink: INKP = ink3dext (the published 3-D ink net) or a channel of CKPT, over the human-label crops (scripts/univ/ink_prep.py:
#         16 crops of 2048 label px on 8 HF ink/phercparis4 segments), rendered on each crop's surface (render --volume), ink-score
#         (F1 / AUC vs the human labels inside the supervision mask)                                                             -> $E/TAG/ink/scores.tsv
# Summary: $E/TAG.matrix.txt (wbench-stats paired CIs against REFS, default "wind96 lat-wz896-12000"; rv / fib / ink rows; xres vs native).
# Env: GPU (1), LKF (a flock file, e.g. /tmp/gpu1-bench.lock, when sharing the GPU with timing runs), UP (predict binary for wbench), U (this tree's build/ufsm), WIN / HALO / WZARGS as
# wbench (latent-grid checkpoints: WZARGS= ), THREADS (8); off the desk: SITE=pod WB=$K/wb K=<kit> (scripts/univ/podkit_eval.sh), INKW / FIBPH paths. Reruns skip finished pieces (wbench rows, rv txt, fib / ink outputs).
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}; TASKS=${TASKS:-wind dlv xres rv fib ink}
H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}; WB=${WB:-/vesuvius/ufsm/wbench}; SITE=${SITE:-desk}; E=${E:-$WB/univ}; RVE=${RVE:-/mnt/scratch/rv/eval}
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; P4K=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
GPU=${GPU:-1}; THREADS=${THREADS:-8}; REFS=${REFS:-wind96 lat-wz896-12000}; DLVS=${DLVS:--1 1}; RVSC=${RVSC:-p1667 p139 p4}
FIBCH=${FIBCH:-}; INKP=${INKP:-}; LKF=${LKF:-}; mkdir -p $E/$T
lk() { if [ -n "$LKF" ]; then flock $LKF "$@"; else "$@"; fi; }   # LKF=/tmp/gpu1-bench.lock: serialise with timing benches
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $E/$T.log; }
wb() {   # wb TAG [env...]: one wbench run (memcap, GPU lock)
  local t=$1; shift; lk env SITE=$SITE WB=$WB GPU=$GPU SLABS=0 THREADS=$THREADS "$@" ~/bin/memcap -m 40G bash $H/scripts/wbench/wbench.sh $CK $t >> $E/$T/wbench_$t.out 2>&1; }
has() { [[ " $TASKS " == *" $1 "* ]]; }
log "univ eval $T ckpt $CK tasks [$TASKS]"

has wind && [ "${WIND:-1}" = 1 ] && { log "wind: wbench core"; wb $T BOXES=core; }
if has dlv && [ "${WIND:-1}" = 1 ]; then for d in $DLVS; do log "dlv $d"; wb $T-dlv$d BOXES=core DLV=$d P4ROOT=$M P4V=$P4K; done; fi
if has xres && [ "${WIND:-1}" = 1 ]; then
  log "xres: native 9.36 um rows"; wb $T-n936 BOXES="139-1 139-2 139-3 139-4 139-5 139-6 139-7 139-8 814-1 814-2 814-3 814-4 814-5 814-6 814-7 814-8 500P2-1"
  for d in 0 -1; do log "xres DLV $d"; wb $T-xres$d BT=$WB/xres/boxes-xres.tsv BOXES=all DLV=$d; done
fi
if has rv && [ "${RV:-1}" = 1 ]; then
  if grep -q "^== $T " $RVE/$T.txt 2>/dev/null && [ $(grep -c ' all ' $RVE/$T.txt) -ge $(echo $RVSC | wc -w) ]; then log "rv: done"; else log "rv: $RVSC"
    # RVU: the rv predict binary (default rv eval.sh's own); latent-grid checkpoints (WZARGS set and empty) get cubic windows (WZ=)
    if [ -n "${WZARGS+x}" ] && [ -z "$WZARGS" ]; then E=$RVE GPU=$GPU U=${RVU:-/vesuvius/ufsm/v2/rv/bin/ufsm.ev} WZ= WINDOW=${WIN:-640} bash $H/scripts/rv/eval.sh $CK $T "$RVSC" >> $E/$T/rv.out 2>&1
    else E=$RVE GPU=$GPU U=${RVU:-/vesuvius/ufsm/v2/rv/bin/ufsm.ev} bash $H/scripts/rv/eval.sh $CK $T "$RVSC" >> $E/$T/rv.out 2>&1; fi; fi
fi
if has fib && [ "${FIB:-1}" = 1 ] && [ -n "$FIBCH" ]; then   # FIBCH e.g. "H=0 V=1 dyad=2-7" (channel indices of CKPT)
  o=$E/$T/fib; mkdir -p $o; BOX=12288,4096,3584,512,576,768; J="--um 2.4 --level 2 --slab 12288,512 --region 4096,4608,3584,4352 --tile 256 --d-mm 0.3,0.6,1.2,2.4"
  for kv in $FIBCH; do n=${kv%%=*}; cs=${kv#*=}; a=${cs%-*}; b=${cs#*-}
    for ch in $(seq $a $b); do d=$o/$n; [ $n = dyad ] && d=$o/dyad/c$ch; [ -e $d/9.6/zarr.json ] && continue
      lk ${UP:-$U} predict $CK $M $P4K $d --um 2.4 --level 2 --box $BOX --channel $ch --gpu $GPU --threads 6 --cache $WB/cache > $o/predict_c$ch.log 2>&1 || log "fib: predict c$ch failed"; done; done
  for f in H V; do [ -e $o/$f/9.6/zarr.json ] || continue; for m in tube dyad; do x=""; [ $m = dyad ] && { [ -d $o/dyad ] || continue; x="--dyad $o/dyad"; }
    [ -s $o/$f-$m.judge ] && continue
    nice -n 10 $U fiber-trace - - $o/$f-$m.fib --mask $o/$f 9.6 --mask-origin 12288,4096,3584 --box $BOX --scaledown 2 --family $f --half 1024 --min-len 48 --threads 4 $x > $o/trace_$f-$m.log 2>&1
    { python3 -I $H/scripts/univ/fiblen.py $o/$f-$m.fib; nice -n 10 $U fiber-judge $o/$f-$m.fib --phase ${FIBPH:-/vesuvius/ufsm/gt/paris4-wind-20261019} phaseA/9.6 $J 2>&1 | grep "$f:"; } > $o/$f-$m.judge
  done; done
fi
if has ink && [ "${INK:-1}" = 1 ] && [ -n "$INKP" ]; then   # INKP = ink3dext | ch:N (channel N of CKPT, ink at 2.4 um level LV_INK (default 1 = 4.8 um))
  o=$E/$T/ink; C=$WB/inkeval/crops; mkdir -p $o; [ -s $o/scores.tsv ] || : > $o/scores.tsv
  grep -v '^#' $C/crops.tsv | while IFS=$'\t' read -r cr seg ns fi box; do
    grep -qP "^$cr\t" $o/scores.tsv && continue
    v=$o/vol; lvi=1
    if [ "$INKP" = ink3dext ]; then lk $U ink3d-ext --weights ${INKW:-/vesuvius/ufsm/teachers/dino3d/ckpt_78k_fullsup.pth} --ct $M $P4K --box $box --out $v --out-um 4.8 --gpu $GPU --cache $WB/cache > $o/ink_$cr.log 2>&1 || { log "ink: $cr predict failed"; continue; }; key=4.8
    else lvi=${LV_INK:-1}; ch=${INKP#ch:}; key=$(python3 -I -c "print(round(2.4 * 2 ** $lvi, 3))")
      lk ${UP:-$U} predict $CK $M $P4K $v --um 2.4 --level $lvi --box $box --channel $ch --gpu $GPU --threads 6 --cache $WB/cache > $o/ink_$cr.log 2>&1 || { log "ink: $cr predict failed"; continue; }; fi
    s=$o/surf/$cr; mkdir -p $s/w; python3 -I - $C/$cr $s/w/$cr $lvi <<'PY'   # the crop surface in the ink volume's voxels (render samples volume voxels)
import sys, os, shutil, numpy as np, tifffile
src, dst, lv = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(dst, exist_ok=True); f = 2.0 ** lv
for a in 'xyz':
    v = tifffile.imread(os.path.join(src, a + '.tif')).astype(np.float32); tifffile.imwrite(os.path.join(dst, a + '.tif'), np.where(v >= 0, (v + 0.5) / f - 0.5, -1).astype(np.float32))
shutil.copy(os.path.join(src, 'meta.json'), dst)
PY
    nice -n 5 $U render $s/w $s/r --volume $v $key --depth 4 --threads 8 --cache $WB/cache > $o/render_$cr.log 2>&1 || { log "ink: $cr render failed"; continue; }
    mp=$(ls $s/r/*.tif 2>/dev/null | grep -v _fill | head -1); [ -n "$mp" ] || { log "ink: $cr no render"; continue; }
    $U ink-score --surface $C/$cr --labels $C/$cr/labels.tif --mask $C/$cr/mask.tif --map ink=$mp --row $cr 2>> $o/score_$cr.log | grep -P "^$cr\t" >> $o/scores.tsv
  done
fi

# ---- matrix ----
{ echo "== univ eval $T  $CK  $(date)"
  for r in $T $(for d in $DLVS; do echo $T-dlv$d; done); do [ -s $WB/res/$r.tsv ] || continue
    for B in $REFS; do echo "-- $r vs $B"; $U wbench-stats $WB/res/$r.tsv $WB/res/$B.tsv --metric p05x,ids 2>&1 | grep -v label-noise | grep -E '^(p05x|ids) '; done; done
  for d in 0 -1; do [ -s $WB/res/$T-xres$d.tsv ] || continue; echo "-- xres DLV $d (fine scan) vs native 9.36 um ($T-n936), same boxes"
    python3 -I $H/scripts/univ/xres_pair.py $WB/res/$T-xres$d.tsv $WB/res/$T-n936.tsv; done
  [ -s $RVE/$T.txt ] && { echo "-- rv (recto-ruler, rv column turns / share off; rv1d2-10000: p1667 0.926/0.377 p139 0.811/0.375 p4 0.738/0.421)"; grep ' all ' $RVE/$T.txt; }
  ls $E/$T/fib/*.judge > /dev/null 2>&1 && { echo "-- fib (HOLD; fibhv-c hold-12000: H tube median 0.57 mm, pairs .96)"; for j in $E/$T/fib/*.judge; do echo "$(basename $j .judge): $(tr '\n' ' ' < $j | cut -c1-260)"; done; }
  [ -s $E/$T/ink/scores.tsv ] && { echo "-- ink (ink-score vs human labels: crop map n inkfrac f1best thr f1_05 prec05 rec05 auc)"; cat $E/$T/ink/scores.tsv
    awk -F'\t' '{n++; f+=$5; a+=$10} END {if (n) printf "mean over %d crops: f1best %.4f auc %.4f\n", n, f / n, a / n}' $E/$T/ink/scores.tsv; }
} > $E/$T.matrix.txt
log "matrix -> $E/$T.matrix.txt"

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
# CTROOT: the CT root for every task but rv (default the volcomp mirror; a tree with the same keys, e.g. a re-encoded sweep arm). Use a fresh E / WB per root: finished pieces are skipped by tag.
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}; TASKS=${TASKS:-wind dlv xres rv fib ink}
H=$(cd "$(dirname "$0")/../.." && pwd); U=${U:-$H/build/ufsm}; WB=${WB:-/vesuvius/ufsm/wbench}; SITE=${SITE:-desk}; E=${E:-$WB/univ}; RVE=${RVE:-/mnt/scratch/rv/eval}
M=${CTROOT:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}; P4K=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
GPU=${GPU:-1}; THREADS=${THREADS:-8}; REFS=${REFS:-wind96 lat-wz896-12000}; DLVS=${DLVS:--1 1}; RVSC=${RVSC:-p1667 p139 p4}
FIBCH=${FIBCH:-}; INKP=${INKP:-}; LKF=${LKF:-}; mkdir -p $E/$T
MC=""; [ -x ~/bin/memcap ] && MC="$HOME/bin/memcap -m 40G"   # (pods have no memcap: the pod cgroup caps them)
lk() { if [ -n "$LKF" ]; then flock $LKF "$@"; else "$@"; fi; }   # LKF=/tmp/gpu1-bench.lock: serialise with timing benches
# lkg: one GPU call of this script (ink / fibre predicts). On the pod's shared GPU ($K/GPU.md) it is one shared unit inside the window;
# wbench and rv wait per predict themselves (site-pod.env RUN, rv/eval.sh), so a whole wbench row is never one unit.
lkg() { if [ -n "${K:-}" ] && [ -d $K/gpu.shared ]; then
    while [ -e $K/gpu.pause ] || [ $((10#$(date +%M))) -ge $([ -e $K/gpu.gate-waiting ] && echo 35 || echo 15) ]; do sleep 5; done
    touch $K/gpu.shared/eval-$$; "$@"; local rc=$?; rm -f $K/gpu.shared/eval-$$; return $rc; else lk "$@"; fi; }
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $E/$T.log; }
wb() {   # wb TAG [env...]: one wbench run (memcap, GPU lock)
  local t=$1; shift; lk env SITE=$SITE WB=$WB GPU=$GPU SLABS=0 THREADS=$THREADS $([ ${PREDONLY:-0} = 1 ] && echo PHASE=predict) "$@" $MC bash $H/scripts/wbench/wbench.sh $CK $t >> $E/$T/wbench_$t.out 2>&1; }
has() { [[ " $TASKS " == *" $1 "* ]]; }
export PREDONLY=${PREDONLY:-0}; [ $PREDONLY = 1 ] && FIB=0
# TWOPHASE=1 (the pod's GPU window, 2026-11-12): first every predict of the eval back to back (this script with PREDONLY=1: wbench
# PHASE=predict, rv predicts, ink predicts; fields and volumes stay on disk), then the plain run, which finds them and only fits / scores.
PO=${PREDONLY:-0}
if [ "${TWOPHASE:-0}" = 1 ] && [ $PO = 0 ]; then tp=$(date +%s); log "gpu phase: all predicts"; PREDONLY=1 bash $0 "$@"; log "gpu phase done in $(( $(date +%s) - tp )) s (queue waits included); fits and scores"
  # the resident trainer's rate during the fits (lead's condition: fits beside U2 must cost it < 10% of 0.72 samp/s / 776 Mvox/s)
  RL=${RESLOG:-/workspace/univA/u2.log}; [ -s $RL ] && wc -l < $RL > $E/$T/reslog.line; fi
[ $PO = 1 ] || log "univ eval $T ckpt $CK tasks [$TASKS]"

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
# latent-grid checkpoints (header "lat_step"): plain predict --channel N / ink skips the fine stem and the ch 0-1 sub-cell shuffle (univI, 2026-11-11),
# so their ink / fibre heads are not scored via --channel until that fix is merged (LATCH=1 overrides). wind (fit fields) and rv (--channel rv) are fine.
LATCK=0; head -c 4096 $CK | grep -aq '"lat_step"' && LATCK=1
if [ $LATCK = 1 ] && [ "${LATCH:-0}" != 1 ]; then
  has fib && [ "${FIBP:-}" != heads ] && { log "fib: skipped, latent checkpoint (plain --channel path not trained-equivalent; FIBP=heads or LATCH=1)"; FIB=0; }
  case "$INKP" in ch:*) log "ink: skipped, latent checkpoint ($INKP via plain --channel; LATCH=1 overrides)"; INK=0;; esac
fi
if has fib && [ "${FIB:-1}" = 1 ] && { [ -n "$FIBCH" ] || [ "${FIBP:-}" = heads ]; }; then   # FIBCH e.g. "H=0 V=1 dyad=2-7" (channel indices of CKPT)
  o=$E/$T/fib; mkdir -p $o; BOX=12288,4096,3584,512,576,768; J="--um 2.4 --level 2 --slab 12288,512 --region 4096,4608,3584,4352 --tile 256 --d-mm 0.3,0.6,1.2,2.4"
  if [ "${FIBP:-}" = heads ]; then   # FIBP=heads (latent-grid checkpoints, 2026-11-12): predict --heads fib writes fib/H, fib/V, fib/dyad/c2..c7
    # from one trunk pass (the trained path: fine stem + sub-cell placement), linked where the --channel path puts them; FIBCH is unused
    [ -e $o/fh/fib/H ] || lkg ${UP:-$U} predict $CK $M $P4K $o/fh --um 2.4 --level 2 --box $BOX --levels 1 --heads fib ${PXH:-} --gpu $GPU --threads 6 --cache $WB/cache > $o/predict_heads.log 2>&1 || log "fib: predict --heads fib failed"
    for n in H V dyad; do [ -e $o/fh/fib/$n ] && ln -sfn fh/fib/$n $o/$n; done; FIBCH=; fi
  for kv in $FIBCH; do n=${kv%%=*}; cs=${kv#*=}; a=${cs%-*}; b=${cs#*-}
    for ch in $(seq $a $b); do d=$o/$n; [ $n = dyad ] && d=$o/dyad/c$ch; [ -e $d/9.6/zarr.json ] && continue
      lkg ${UP:-$U} predict $CK $M $P4K $d --um 2.4 --level 2 --box $BOX --channel $ch --gpu $GPU --threads 6 --cache $WB/cache > $o/predict_c$ch.log 2>&1 || log "fib: predict c$ch failed"; done; done
  for f in H V; do [ -e $o/$f/9.6/zarr.json ] || continue; for m in tube dyad; do x=""; [ $m = dyad ] && { [ -d $o/dyad ] || continue; x="--dyad $o/dyad"; }
    [ -s $o/$f-$m.judge ] && continue
    # (eval audit 11-14 #5: the judge's GT phaseA was deleted on 11-12, and the old line wrote fiblen's output even when fiber-judge failed, so a
    #  missing GT left a non-empty .judge that later runs skipped as done. The .judge file is now written only when the trace and the judge
    #  both succeed and the judge printed its family line; the GT is phaseB (the july set, the only Paris 4 phase left; rows before 11-14
    #  were judged against phaseA and are not comparable). FIBPH / FIBPHK override.)
    ph=${FIBPH:-/vesuvius/ufsm/gt/paris4-wind-20261019}; phk=${FIBPHK:-phaseB/9.6}
    [ -e $ph/$phk/zarr.json ] || { log "fib: FAILED, no GT phase $ph/$phk"; FIBFAIL=1; continue; }
    nice -n 10 $U fiber-trace - - $o/$f-$m.fib --mask $o/$f 9.6 --mask-origin 12288,4096,3584 --box $BOX --scaledown 2 --family $f --half 1024 --min-len 48 --threads 4 $x > $o/trace_$f-$m.log 2>&1 ||
      { log "fib: FAILED, fiber-trace $f-$m (see $o/trace_$f-$m.log)"; FIBFAIL=1; continue; }
    jl=$(nice -n 10 $U fiber-judge $o/$f-$m.fib --phase $ph $phk $J 2>&1 | tee $o/judge_$f-$m.log | grep "$f:")
    [ -n "$jl" ] || { log "fib: FAILED, fiber-judge $f-$m printed no $f: line (see $o/judge_$f-$m.log)"; FIBFAIL=1; continue; }
    { python3 -I $H/scripts/univ/fiblen.py $o/$f-$m.fib; echo "$jl"; echo "gt $ph $phk"; } > $o/$f-$m.judge.tmp && mv $o/$f-$m.judge.tmp $o/$f-$m.judge
  done; done
fi
if has ink && [ "${INK:-1}" = 1 ] && [ -n "$INKP" ]; then   # INKP = ink3dext | heads (predict --heads ink: latent nets) | ch:N (channel N of CKPT, ink at 2.4 um level LV_INK (default 1 = 4.8 um))
  # crop sets: the 16 Paris 4 crops ($WB/inkeval/crops) + INKC (inkB's held-out 0139 / 1667 crops; crops.tsv cols 6-7 = scroll, CT key; um in crop.json).
  # INKGT: 3-D ink-splat references <crop>.zarr (inkB, ink-splat --face): each crop is also scored voxelwise (ink-vscore -> vscores.tsv).
  # crop boxes are level-0 voxels; predict --box is in voxels of --level, and --full-grid 1 writes the store in whole-volume coordinates
  # (what ink-vscore and the crop surfaces index). Without both, the 2026-11-12 smoke test read air and wrote a box-relative store.
  lbox() { python3 -I -c "import sys; b = [int(v) for v in sys.argv[1].split(',')]; f = 1 << int(sys.argv[2]); print(','.join(str(v) for v in [b[0] // f, b[1] // f, b[2] // f, -(-(b[0] + b[3]) // f) - b[0] // f, -(-(b[1] + b[4]) // f) - b[1] // f, -(-(b[2] + b[5]) // f) - b[2] // f]))" $1 $2; }
  o=$E/$T/ink; mkdir -p $o; [ -s $o/scores.tsv ] || : > $o/scores.tsv; [ -s $o/vscores.tsv ] || : > $o/vscores.tsv
  for C in $WB/inkeval/crops ${INKC:-}; do [ -s $C/crops.tsv ] || continue
  grep -v '^#' $C/crops.tsv | while IFS=$'\t' read -r cr seg ns fi box scr ctk; do
    [ -n "${INKONLY:-}" ] && [ "$cr" != "$INKONLY" ] && continue   # INKONLY=CROP: that crop alone (a caller yielding GPU 0 between crops)
    ctk=${ctk:-$P4K}; case $M in http*) ;; *) ctk=${ctk/\/volumes\//\/};; esac   # (a local CT tree, e.g. the desk's /vesuvius/ufsm/v2/ct: keys without volumes/)
    um=$(python3 -I -c "import json; print(json.load(open('$C/$cr/crop.json')).get('um', 2.4))")
    g=/nonexistent; for gd in ${INKGT:-}; do [ -d $gd/$cr.zarr ] && { g=$gd/$cr.zarr; break; }; done; vs=0; [ -d $g ] && ! grep -qP "^$cr\t" $o/vscores.tsv && vs=1   # INKGT: one or more dirs
    grep -qP "^$cr\t" $o/scores.tsv && [ $vs = 0 ] && continue
    v=$o/vol_${scr:-PHercParis4}; [ "$INKP" = heads ] && v=$o/vol_$cr   # (predict --heads has no --resume: a store per crop)
    if [ "$INKP" = ink3dext ]; then lvi=1; key=$(python3 -I -c "print(round($um * 2, 3))")
      [ -e $o/pred_$cr.done ] || lkg $U ink3d-ext --model dino3d --weights ${INKW:-/vesuvius/ufsm/teachers/dino3d/ckpt_78k_fullsup.pth} --ct $M $ctk/0 --ct-um $um --box $box --surface $C/$cr --out $v --out-um 4.8 --gpu $GPU --cache $WB/cache > $o/ink_$cr.log 2>&1 || { log "ink: $cr predict failed"; continue; }
    elif [ "$INKP" = heads ]; then lvi=${LV_INK:-1}; key=$(python3 -I -c "print(round($um * 2 ** $lvi, 3))")   # predict --heads ink: the latent-grid
      # multi-head path (fine stem rows + sub-cell placement as trained; the ink channel from the header), written to $v.h/ink, read through $v
      [ -e $v ] || ln -sfn $(basename $v).h/ink $v
      if [ "${BLEND:-0}" = 1 ]; then   # BLEND=1 (lead 2026-11-12): the latent ink head's level steps at window seams (inkB: 10-30 / 255, halo 16..256
        # alike), so two passes on tile grids offset by half the 512 core (--grid-origin, in level-LV voxels) blended with sin^2 weights, as
        # inkB's sched/inkwatch/u1667ink-v2.sh (ink-blend: seam step 16.9 -> 1.3 on a 1299 box); the passes are kept in $v.g0 / $v.g256
        [ -e $o/pred_$cr.done ] || { for gg in 0 256; do [ -e $v.g$gg/ink/zarr.json ] || lkg ${UP:-$U} predict $CK $M $ctk $v.g$gg --um $um --level $lvi --box $(lbox $box $lvi) --full-grid 1 --levels 1 --heads ink --window 544 --halo 16 --grid-origin $gg,$gg,$gg ${PXH:-} --gpu $GPU --threads 6 --cache $WB/cache > $o/ink_${cr}_g$gg.log 2>&1 || break; done
          [ -e $v.g0/ink/zarr.json ] && [ -e $v.g256/ink/zarr.json ] && mkdir -p $v.h && ${UP:-$U} ink-blend $v.h/ink --um $key --tile 512,512,512 --grids 0,0,0:256,256,256 $v.g0/ink $v.g256/ink --threads 8 > $o/blend_$cr.log 2>&1 && [ -e $v.h/ink/zarr.json ]; } || { log "ink: $cr blended predict failed"; continue; }
      else
      [ -e $o/pred_$cr.done ] || lkg ${UP:-$U} predict $CK $M $ctk $v.h --um $um --level $lvi --box $(lbox $box $lvi) --full-grid 1 --levels 1 --heads ink ${PXH:-} --gpu $GPU --threads 6 --cache $WB/cache > $o/ink_$cr.log 2>&1 || { log "ink: $cr predict failed"; continue; }; fi
    else lvi=${LV_INK:-1}; ch=${INKP#ch:}; key=$(python3 -I -c "print(round($um * 2 ** $lvi, 3))")
      [ -e $o/pred_$cr.done ] || lkg ${UP:-$U} predict $CK $M $ctk $v --um $um --level $lvi --box $(lbox $box $lvi) --full-grid 1 --levels 1 --resume 1 --channel $ch --gpu $GPU --threads 6 --cache $WB/cache > $o/ink_$cr.log 2>&1 || { log "ink: $cr predict failed"; continue; }; fi
    touch $o/pred_$cr.done; [ $PO = 1 ] && continue
    if [ $vs = 1 ]; then gk=$(ls $g | grep -vE '^(segments|zarr.json)$' | head -1)
      $U ink-vscore --gt $g $gk --pred $v $key --box $box --level $lvi --row $cr --threads 8 2>> $o/vscore_$cr.log | grep -P "^$cr\t" >> $o/vscores.tsv; fi
    grep -qP "^$cr\t" $o/scores.tsv && continue
    s=$o/surf/$cr; mkdir -p $s/w; python3 -I - $C/$cr $s/w/w_$cr $lvi <<'PY'   # the crop surface in the ink volume's voxels (render samples volume voxels; render only reads wrap dirs named w*)
import sys, os, shutil, numpy as np, tifffile
src, dst, lv = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(dst, exist_ok=True); f = 2.0 ** lv
for a in 'xyz':
    v = tifffile.imread(os.path.join(src, a + '.tif')).astype(np.float32); tifffile.imwrite(os.path.join(dst, a + '.tif'), np.where(v >= 0, (v + 0.5) / f - 0.5, -1).astype(np.float32))
shutil.copy(os.path.join(src, 'meta.json'), dst)
PY
    lkg nice -n 5 $U render $s/w $s/r --volume $v $key --depth 4 --threads 8 --cache $WB/cache > $o/render_$cr.log 2>&1 || { log "ink: $cr render failed"; continue; }
    mp=$(ls $s/r/*.tif 2>/dev/null | grep -v _fill | head -1); [ -n "$mp" ] || { log "ink: $cr no render"; continue; }
    $U ink-score --surface $C/$cr --labels $C/$cr/labels.tif --mask $C/$cr/mask.tif --map ink=$mp --row $cr 2>> $o/score_$cr.log | grep -P "^$cr\t" >> $o/scores.tsv
  done; done
fi

[ $PO = 1 ] && exit $([ "${FIBFAIL:-0}" = 0 ] && echo 0 || echo 1)
# ---- matrix ----
{ echo "== univ eval $T  $CK  $(date)"
  ws() { $U wbench-stats $1 $2 --metric p05x,ids 2>&1 | grep -v label-noise | grep -E '^(p05x|ids) '; }
  [ -s $WB/res/$T.tsv ] && for B in $REFS; do [ -s $WB/res/$B.tsv ] && { echo "-- $T vs $B (native level)"; ws $WB/res/$T.tsv $WB/res/$B.tsv; }; done
  # fed levels (DLV): like with like (the reference fed the same level; a reference's native row is NOT the comparison: every net drops
  # off its native level), then this net's own drop from its native row
  for d in $DLVS; do r=$T-dlv$d; [ -s $WB/res/$r.tsv ] || continue
    for B in $REFS; do [ -s $WB/res/$B-dlv$d.tsv ] && { echo "-- $r vs $B-dlv$d (both fed DLV $d)"; ws $WB/res/$r.tsv $WB/res/$B-dlv$d.tsv; }; done
    [ -s $WB/res/$T.tsv ] && { echo "-- $r vs $T (this net: fed DLV $d - native)"; ws $WB/res/$r.tsv $WB/res/$T.tsv; }; done
  # masked-holdout control (scripts/univ/holdsplit.py): A - B on the boxes U1 saw the CT of (labels masked) vs the reject boxes (CT unseen)
  REJ=${REJ:-$H/scripts/univ/u1reject.txt}
  [ -s $REJ ] && [ -s $WB/res/$T.tsv ] && for B in $REFS; do [ -s $WB/res/$B.tsv ] || continue; pr="$WB/res/$T.tsv:$WB/res/$B.tsv"
    [ -s $WB/res/$T-n936.tsv ] && [ -s $WB/res/$B-n936.tsv ] && pr="$pr $WB/res/$T-n936.tsv:$WB/res/$B-n936.tsv"
    echo "-- masked vs reject boxes, $T - $B (core + native 9.36 um rows; reject = $(grep -v '^#' $REJ | tr '\n' ' '))"; python3 -I $H/scripts/univ/holdsplit.py $REJ $pr
    # the same split for a snapshot trained before any masking (REJREF, u1-3000-prefix: every holdout window rejected then). Its masked - reject
    # is the box mix alone (0139 + HOLD vs Paris 4 / 1667: +0.018 p05x, +0.049 ids vs lat-wz896p), so the CT effect is this net's value minus that one.
    R0=${REJREF:-u1-3000-prefix}; [ $R0 != $T ] && [ -s $WB/res/$R0.tsv ] && { pr="$WB/res/$R0.tsv:$WB/res/$B.tsv"
      [ -s $WB/res/$R0-n936.tsv ] && [ -s $WB/res/$B-n936.tsv ] && pr="$pr $WB/res/$R0-n936.tsv:$WB/res/$B-n936.tsv"
      echo "-- the same split before masking, $R0 - $B (the box-mix baseline to subtract)"; python3 -I $H/scripts/univ/holdsplit.py $REJ $pr | grep 'masked - reject'; }; done
  for d in 0 -1; do [ -s $WB/res/$T-xres$d.tsv ] || continue; echo "-- xres DLV $d (fine scan) vs native 9.36 um ($T-n936), same boxes"
    python3 -I $H/scripts/univ/xres_pair.py $WB/res/$T-xres$d.tsv $WB/res/$T-n936.tsv; done
  [ -s $RVE/$T.txt ] && { echo "-- rv (recto-ruler; each cell = turns per GT turn (1 = right, < 1 under-counts) / share of GT intervals off >= 0.5 turn (lower is better))"; echo "   columns: scroll all n | fit (the reference count fit, not the model) | density | CT | rv.pkw | rv.pk (the model; the compared column) | rv.run"; echo "   rv1d2-10000 rv.pk: p1667 0.926/0.377 p139 0.811/0.375 p4 0.738/0.421; rv.run: p1667 1.135/0.384 p139 1.002/0.398 p4 1.005/0.274"; grep ' all ' $RVE/$T.txt; }
  ls $E/$T/fib/*.judge > /dev/null 2>&1 && { echo "-- fib (HOLD; fibhv-c hold-12000: H tube median 0.57 mm, pairs .96)"; for j in $E/$T/fib/*.judge; do echo "$(basename $j .judge): $(tr '\n' ' ' < $j | cut -c1-260)"; done; }
  [ -s $E/$T/ink/scores.tsv ] && { echo "-- ink (ink-score vs human labels: crop map n inkfrac f1best thr f1_05 prec05 rec05 auc)"; cat $E/$T/ink/scores.tsv
    awk -F'\t' '{n++; f+=$5; a+=$10} END {if (n) printf "mean over %d crops: f1best %.4f auc %.4f\n", n, f / n, a / n}' $E/$T/ink/scores.tsv; }
    [ -s $E/$T/ink/vscores.tsv ] && echo "-- ink voxelwise (ink-vscore vs 3-D ink-splat refs: crop P N known_frac ink_share auc thr f1best f1_05 prec05 rec05)" && cat $E/$T/ink/vscores.tsv && awk -F"\t" "{n++; a+=\$6; f+=\$8} END {if (n) printf \"voxelwise vs 3-D refs over %d crops: auc %.4f f1best %.4f\\n\", n, a / n, f / n}" $E/$T/ink/vscores.tsv;
    # per scroll (crop names: w.. = Paris 4, else SCROLL_w..): held-out rows named for what the trunk saw (INKSEEN "SCROLL=label ...")
    for f in scores:5:10 vscores:8:6; do IFS=: read -r fn fc ac <<< "$f"; [ -s $E/$T/ink/$fn.tsv ] || continue
      awk -F'\t' -v fc=$fc -v ac=$ac -v lab="${INKSEEN:-PHerc0500P2=unseen-ink-labels,seen-CT PHerc0841=unseen-scroll}" -v w=$fn 'BEGIN {n = split(lab, L, " "); for (i = 1; i <= n; i++) { split(L[i], kv, "="); D[kv[1]] = kv[2] } }
        { s = ($1 ~ /^w[0-9]/) ? "PHercParis4" : substr($1, 1, index($1, "_w") - 1); c[s]++; F[s] += $fc; A[s] += $ac }
        END { for (s in c) printf "%s per scroll %-12s %-28s n %2d auc %.4f f1best %.4f\n", w, s, (s in D) ? D[s] : "", c[s], A[s] / c[s], F[s] / c[s] }' $E/$T/ink/$fn.tsv | sort; done
  if [ -s $E/$T/reslog.line ] && [ -s ${RESLOG:-/workspace/univA/u2.log} ]; then   # (step lines: '... X samp/s Y Mvox/s')
    tail -n +$(( $(cat $E/$T/reslog.line) + 1 )) ${RESLOG:-/workspace/univA/u2.log} | grep -E '^ *step [0-9]+ .*samp/s' | awk '{ for (i = 2; i <= NF; i++) { if ($i == "samp/s") s = $(i - 1); if ($i == "Mvox/s") v = $(i - 1) } n++; S += s; V += v }
      END { if (n) printf "-- resident trainer during the fits: %d step lines, %.3f samp/s (%.1f%% of 0.72), %.0f Mvox/s (%.1f%% of 776)\n", n, S / n, 100 * S / n / 0.72, V / n, 100 * V / n / 776 }'; fi
} > $E/$T.matrix.txt
log "matrix -> $E/$T.matrix.txt"
[ "${FIBFAIL:-0}" = 0 ] || { log "fib: at least one judge FAILED (no .judge written for it)"; exit 1; }

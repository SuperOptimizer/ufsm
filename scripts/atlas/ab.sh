#!/bin/bash
# atlas recto-detection A/B (2026-11-27, atlas; lead: card 0 slot n3, after /vesuvius/ufsm/lead/card0.2.done). From try9cool cooled-33333 (the
# production rv head, unwrap2 --rv 2), try9's flags / seed 3 / STEPS (80) per arm:
#   A = try9's sources (sources-A.json), B = + 21 gated ATLAS sources at the base's total weight (half the draws; heads rv + normal).
# Then predict --heads rv (base, A, B) and atlas-mini-h (the boxes it has not run yet) on 7 held GT boxes (139-5, 139-2, Paris4 HOLD,
# 1667 C2 / C10 / C20 / C23; C20 / C23 / 139-2 the wbench packed picks) and 3 held ATLAS slabs (0800, 0306B x2), all scored on the CPU by
# ufsm atlas-rvscore (side accuracy, surface F1 within 2 voxels vs GT, ATLAS recto-face orientation and node recall). score.tsv.
# Card 0: gpu0.excl exclusive (turns: flock -w 5 -E 75, retry) + score.lock.0 for the GPU part; DONE touched after it.
# Launch: setsid nohup bash /vesuvius/ufsm/atlas/ab/ab.sh > /vesuvius/ufsm/atlas/ab/ab.log 2>&1 < /dev/null &
set -u
D=/vesuvius/ufsm/atlas/ab; A0=/vesuvius/ufsm/atlas; MI=$A0/mini; U=/vesuvius/ufsm/bin/ufsm-atlas; CTR=/vesuvius/ufsm/v2/ct; G=/vesuvius/ufsm/gt
GO=${GO:-/vesuvius/ufsm/lead/card0.2.done}; DONE=${DONE:-/vesuvius/ufsm/lead/card0.3-atlas.done}; STEPS=${STEPS:-80}; LK=/tmp/ufsm-suite-locks
BASE=/vesuvius/ufsm/v2/lat/runs/try9cool/cooled-33333.ckpt
say() { echo "$(date '+%F %T') $*" | tee -a $D/ab.txt; }
cd $D
say "waiting for $GO"
while [ ! -e $GO ]; do sleep 30; done
say "go; locking card 0"
exec 8> $LK/gpu0.excl; until flock -x -w 5 -E 75 8; do sleep 60; done
exec 9> /vesuvius/ufsm/inksdk/score.lock.0; until flock -x -w 5 -E 75 9; do sleep 60; done
say "card 0 locked"
# ---- the two arms
for arm in A B; do
  OUT=$D/runs/$arm; [ -e $OUT.done ] && continue; mkdir -p $OUT
  SRCJ=$D/sources-$arm.json; ECHOMAX=4; GPUAUG=3; STICK=8; . $D/args.sh
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 UFSM_SAMPLER_THP=0 UFSM_NCPU=${NCPU:-16} timeout -k 30 $((STEPS * 60 + 900)) ~/bin/memcap -t -m 80G $U train "${A[@]}" > $OUT/train.log 2>&1
  rc=$?; say "train $arm rc $rc $(( $(date +%s) - t0 )) s, DEFAULTED $(grep -c DEFAULTED $OUT/train.log), last: $(ls $OUT/*.ckpt 2> /dev/null | sort -V | tail -1)"
  [ $rc = 0 ] && touch $OUT.done
done
ck() { case $1 in base) echo $BASE;; *) ls $D/runs/$1/*.ckpt 2> /dev/null | grep -v ema | sort -V | tail -1;; esac; }
# name | ct key (group) | um | level | box (level voxels) | axis | axis-div | kind (scoring)
cat > $D/boxes.tsv << EOF
139-5	PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr	9.362	0	5376,3072,3840,512,512,512	/vesuvius/ufsm/gt/axis/PHerc0139-umb-scan.json	1	g139
139-2	PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr	9.362	0	5376,4096,2560,512,512,512	/vesuvius/ufsm/gt/axis/PHerc0139-umb-scan.json	1	g139
HOLD	PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr	2.4	2	12288,4096,3584,512,576,768	/vesuvius/ufsm/teacher/axis-0.json	4	gp4
C2	PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr	2.399	2	3328,1152,512,512,512,512	/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json	4	g1667
C10	PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr	2.399	2	8448,2432,2048,512,512,512	/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json	4	g1667
C20	PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr	2.399	2	7680,512,2048,512,512,512	/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json	4	g1667
C23	PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr	2.399	2	5504,512,2432,512,512,512	/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json	4	g1667
A0800	PHerc0800/20250521135224-8.640um-1.2m-116keV-masked.zarr	8.64	0	11150,4472,4472,634,784,785	/vesuvius/ufsm/pods/PHerc0800/unroll/axis.json	1	eval-ids-PHerc0800_4_4.txt
A0306a	PHerc0306B/20250521133212-8.640um-1.2m-116keV-masked.zarr	8.64	0	8824,4344,3704,785,785,784	/vesuvius/ufsm/scratch-mig/pods/PHerc0306B/run/axis.json	1	eval-ids-PHerc0306B_3_4.txt
A0306b	PHerc0306B/20250521133212-8.640um-1.2m-116keV-masked.zarr	8.64	0	8824,5368,4728,784,739,784	/vesuvius/ufsm/scratch-mig/pods/PHerc0306B/run/axis.json	1	eval-ids-PHerc0306B_4_5.txt
EOF
for arm in base A B; do
  c=$(ck $arm); [ -n "$c" ] || { say "no checkpoint for $arm"; continue; }
  while IFS=$'\t' read b k um lv box ax adiv kind; do
    o=$D/pred/$arm/$b; [ -e $o.ok ] && continue; mkdir -p $D/pred/$arm
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 timeout -k 30 1800 ~/bin/memcap -m 24G $U predict $c $CTR $k $o --box $box --um $um --level $lv --full-grid 1 \
      --window 640 --halo 64 --shard 128 --threads 8 --gpu 0 --axis $ax --heads rv > $o.log 2>&1
    rc=$?; [ $rc = 0 ] && touch $o.ok; say "predict $arm $b rc $rc $(( $(date +%s) - t0 )) s"
  done < $D/boxes.tsv
done
# ---- atlas-mini-h on the boxes it has not run yet (measurement only; local pinned weights, offline)
mb=""; while IFS=$'\t' read b rest; do [ -s $MI/out/${b}_m16.npy ] || mb="$mb $b"; done < $D/boxes.tsv
if [ -n "$mb" ]; then
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 timeout -k 30 3600 ~/bin/memcap -m 24G $MI/venv/bin/python -I $MI/run_pred.py box $mb > $D/mini.log 2>&1
  say "atlas-mini-h$mb rc $? $(( $(date +%s) - t0 )) s"
fi
flock -u 9; flock -u 8; exec 9>&- 8>&-
touch $DONE; say "card 0 released ($DONE)"
# ---- CPU scoring: one table (score.tsv)
S=$D/score.tsv; : > $S
sc() { UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore "$@" --out $S 2>> $D/score.err; }
for arm in base A B mini; do
  while IFS=$'\t' read b k um lv box ax adiv kind; do
    read z y x nz ny nx < <(echo $box | tr , ' ')
    if [ $arm = mini ]; then [ -s $MI/out/${b}_m16.npy ] || continue; P=(--prob $MI/out/${b}_m16.npy --prob-origin $((z - 16)),$((y - 16)),$((x - 16)))
    else o=$D/pred/$arm/$b; [ -e $o.ok ] || continue; P=(--rv $o rv/$(ls $o/rv | sort -g | head -1)); fi
    case $kind in
      g139)  X=(--box $box --phase $G PHerc0139-wind-20261016/phase/18.72 --phase-div 2 --codes $G PHerc0139-wind-20261016/codes/18.72 --codes-scale 2);;
      gp4)   X=(--box $box --phase $G paris4-phaseA-evalboxes-20261114/phaseA/9.6 --codes $G paris4-q9p6/q-9p6.zarr/9.6 --codes-scale 1);;
      g1667) X=(--box $box --phase $MI/phase1667 9.6 --codes $G PHerc1667-wind-20261016/codes/4.798 --codes-scale 0.5);;
      *)     X=(--patches $A0/surf/index.tsv --ids $A0/$kind);;
    esac
    [ $arm != mini ] && case $kind in g*) X+=(--ct $CTR $k/$lv);; esac
    sc "${P[@]}" "${X[@]}" --axis $ax --axis-div $adiv --tag $arm:$b; say "score $arm $b rc $?"
  done < $D/boxes.tsv
done
say "SCORED: $S"

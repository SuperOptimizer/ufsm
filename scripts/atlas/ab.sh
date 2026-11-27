#!/bin/bash
# atlas recto-detection A/B (2026-11-27, atlas): fine-tunes from try9cool cooled-33333 (the production rv head, --rv 2), same flags / seed / steps:
#   A = try9's sources (sources-A.json = sources-scratch1-ph80-ho), B = + 21 gated ATLAS sources (weight 0.148, heads rv + normal; sources-B.json).
# Then predict --heads rv (base, A, B) on 4 GT boxes (139-5, Paris4 HOLD, 1667 C2 / C10: every one held out of A and B) and 3 held ATLAS slabs
# (0800 / 0306B, holdouts-atlas.json), scored on the CPU with atlas-rvscore (GT phase sides / ATLAS recto-face orientation). results.tsv.
# Card 0 by the lead's slot: waits for $GO, then holds gpu0.excl (exclusive) + score.lock.0 for the whole GPU part; touches $DONE after it.
# Launch: setsid nohup bash /vesuvius/ufsm/atlas/ab/ab.sh > /vesuvius/ufsm/atlas/ab/ab.log 2>&1 < /dev/null &
set -u
D=/vesuvius/ufsm/atlas/ab; A0=/vesuvius/ufsm/atlas; U=/vesuvius/ufsm/bin/ufsm-atlas; CTR=/vesuvius/ufsm/v2/ct; G=/vesuvius/ufsm/gt
GO=${GO:-$D/go}; DONE=${DONE:-/vesuvius/ufsm/lead/card0.atlas.done}; STEPS=${STEPS:-80}; LK=/tmp/ufsm-suite-locks
BASE=/vesuvius/ufsm/v2/lat/runs/try9cool/cooled-33333.ckpt
say() { echo "$(date '+%F %T') $*" | tee -a $D/ab.txt; }
cd $D
while [ ! -e $GO ]; do sleep 30; done
say "go; locking card 0"
exec 8> $LK/gpu0.excl; until flock -x -w 5 -E 75 8; do say "gpu0.excl busy, retry in 60 s"; sleep 60; done
exec 9> /vesuvius/ufsm/inksdk/score.lock.0; until flock -x -w 5 -E 75 9; do say "score.lock.0 busy, retry in 60 s"; sleep 60; done
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
# ---- predictions: name | ct key | um | level | box (level voxels) | axis
cat > $D/boxes.tsv << EOF
139-5	PHerc0139/20250728140407-9.362um-1.2m-113keV-masked.zarr	9.362	0	5376,3072,3840,512,512,512	/vesuvius/ufsm/gt/axis/PHerc0139-umb-scan.json
HOLD	PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr	2.4	2	12288,4096,3584,512,576,768	/vesuvius/ufsm/teacher/axis-0.json
C2	PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr	2.399	2	3328,1152,512,512,512,512	/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json
C10	PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr	2.399	2	8448,2432,2048,512,512,512	/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json
A0800	PHerc0800/20250521135224-8.640um-1.2m-116keV-masked.zarr	8.64	0	11150,4472,4472,634,784,785	/vesuvius/ufsm/pods/PHerc0800/unroll/axis.json
A0306a	PHerc0306B/20250521133212-8.640um-1.2m-116keV-masked.zarr	8.64	0	8824,4344,3704,785,785,784	/vesuvius/ufsm/scratch-mig/pods/PHerc0306B/run/axis.json
A0306b	PHerc0306B/20250521133212-8.640um-1.2m-116keV-masked.zarr	8.64	0	8824,5368,4728,784,739,784	/vesuvius/ufsm/scratch-mig/pods/PHerc0306B/run/axis.json
EOF
for arm in base A B; do
  c=$(ck $arm); [ -n "$c" ] || { say "no checkpoint for $arm"; continue; }
  while IFS=$'\t' read b k um lv box ax; do
    o=$D/pred/$arm/$b; [ -e $o.ok ] && continue; mkdir -p $D/pred/$arm
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 timeout -k 30 1800 ~/bin/memcap -m 24G $U predict $c $CTR $k $o --box $box --um $um --level $lv --full-grid 1 \
      --window 640 --halo 64 --shard 128 --threads 8 --gpu 0 --axis $ax --heads rv > $o.log 2>&1
    rc=$?; [ $rc = 0 ] && touch $o.ok; say "predict $arm $b rc $rc $(( $(date +%s) - t0 )) s"
  done < $D/boxes.tsv
done
flock -u 9; flock -u 8; exec 9>&- 8>&-
touch $DONE; say "card 0 released ($DONE)"
# ---- CPU scoring
S=$D/score.tsv; : > $S
for arm in base A B; do
  while IFS=$'\t' read b k um lv box ax; do
    o=$D/pred/$arm/$b; [ -e $o.ok ] || continue; key=rv/$(ls $o/rv | sort -g | head -1)
    case $b in
      139-5) UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore --rv $o $key --box $box --phase $G PHerc0139-wind-20261016/phase/18.72 --phase-div 2 --ct $CTR $k/0 --tag $arm:$b --out $S;;
      HOLD)  UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore --rv $o $key --box $box --phase $G paris4-phaseA-evalboxes-20261114/phaseA/9.6 --ct $CTR $k/2 --tag $arm:$b --out $S;;
      C2|C10) UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore --rv $o $key --box $box --phase $A0/mini/phase1667 9.6 --ct $CTR $k/2 --tag $arm:$b --out $S;;
      A0800) UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore --rv $o $key --patches $A0/surf/index.tsv --ids $A0/eval-ids-PHerc0800_4_4.txt --axis $ax --tag $arm:$b --out $S;;
      A0306a) UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore --rv $o $key --patches $A0/surf/index.tsv --ids $A0/eval-ids-PHerc0306B_3_4.txt --axis $ax --tag $arm:$b --out $S;;
      A0306b) UFSM_NCPU=6 ~/bin/memcap -m 16G $U atlas-rvscore --rv $o $key --patches $A0/surf/index.tsv --ids $A0/eval-ids-PHerc0306B_4_5.txt --axis $ax --tag $arm:$b --out $S;;
    esac
    say "score $arm $b rc $?"
  done < $D/boxes.tsv
done
say "SCORED: $S"

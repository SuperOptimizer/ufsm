#!/usr/bin/env bash
# scripts/fibers/evald.sh STEP (pod 2): the fibhv-d hold evaluation of one snapshot. The snapshot comes from the training pod (TP, ssh key
# KEY); all 8 output channels (H, V masks; dyad c2..c7) are predicted on the GPU over
#   HOLD       Paris 4 2.4 um scan, level 2 (9.6 um), level-2 box 12288,4096,3584,512,576,768 (wbench holdout; GT judge on the desk)
#   1447-HOLD  PHerc1447 8.64 um, level 0, box 5120,3840,3072,512,768,768 (fibhv-d holdout; fibhv-a..c saw it)
# then each family is traced tube-only and with --dyad (fiber-trace --mask, as fibhv-c.eval12000.sh: min length 0.46 mm, half 1024
# prediction voxels), and on 1447-HOLD the masks are scored against the held-out label raster (ufsm eval, labels copied from TP).
# Output /workspace/fibd/eval/hold-STEP/{HOLD,1447}/...; touch .../done. The desk (evald_desk.sh) copies HOLD/*.fib back for the GT judge.
set -u
s=${1:?step}; TP=${TP:-root@213.173.104.45}; TPP=${TPP:-40689}; KEY=${KEY:-/root/.ssh/id_fibd}; RUN=${RUN:-/workspace/fibers/runs/fibhv-d}
U=${U:-/workspace/ufsm-fibd/build/ufsm}; CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp; C=/workspace/cache; o=/workspace/fibd/eval/hold-$s
E="ssh -p $TPP -i $KEY"; mkdir -p $o/HOLD $o/1447 /workspace/fibd/snap
until $E $TP "test -s $RUN/snap/$s.ckpt" </dev/null; do sleep 120; done; sleep 30
ck=/workspace/fibd/snap/$s.ckpt; [ -s $ck ] || rsync -a -e "$E" $TP:$RUN/snap/$s.ckpt $ck
pred() {   # pred DIR KEY UM LEVEL BOX
  for ch in 0 1 2 3 4 5 6 7; do d=$1/dyad/c$ch; [ $ch = 0 ] && d=$1/H; [ $ch = 1 ] && d=$1/V
    [ -e $d/zarr.json ] || $U predict $ck $CT $2 $d --um $3 --level $4 --box $5 --channel $ch --gpu 0 --threads 8 --cache $C > $1/predict_c$ch.log 2>&1 || echo "predict c$ch failed ($1)"
  done; }
trace() {   # trace DIR MASKKEY ORIGIN BOX SCALEDOWN MINLEN
  for f in H V; do for m in tube dyad; do x=""; [ $m = dyad ] && x="--dyad $1/dyad"
    nice -n 10 $U fiber-trace - - $1/$f-$m.fib --mask $1/$f $2 --mask-origin $3 --box $4 --scaledown $5 --family $f --half 1024 --min-len $6 --threads 12 $x > $1/trace_$f-$m.log 2>&1 || echo "trace $f-$m failed ($1)"
  done; done; }
pred $o/HOLD PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr 2.4 2 12288,4096,3584,512,576,768
pred $o/1447 PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr 8.64 0 5120,3840,3072,512,768,768
trace $o/HOLD 9.6 12288,4096,3584 12288,4096,3584,512,576,768 2 48
trace $o/1447 8.64 5120,3840,3072 5120,3840,3072,512,768,768 0 53
# 1447-HOLD masks against the held-out ring labels (shards z 10, y 7..8, x 6..7 of the 512^3 label grid)
L=/workspace/fibd/lab1447; for f in H V; do mkdir -p $L/$f/8.64/c/10
  rsync -a -e "$E" $TP:/workspace/fibers/labels2/PHerc1447/$f/zarr.json $L/$f/ ; rsync -a -e "$E" $TP:/workspace/fibers/labels2/PHerc1447/$f/8.64/zarr.json $L/$f/8.64/
  rsync -a -e "$E" $TP:/workspace/fibers/labels2/PHerc1447/$f/8.64/c/10/ $L/$f/8.64/c/10/
  $U eval $o/1447/$f . $L/$f . --um 8.64 --box 5120,3840,3072,512,768,768 --pred-origin 5120,3840,3072 --thr 0.3,0.5,0.7 > $o/1447/eval_$f.txt 2>&1
done
touch $o/done

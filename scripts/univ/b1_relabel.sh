#!/usr/bin/env bash
# scripts/univ/b1_relabel.sh SET: workstream B1 of the universal-net plan - winding phase labels at 4.8 um (level 1 of a 2.4 um scan)
# from the published segments, with the GPU pair rule (phase-raster --gpu, labelfast). The 9.6 um sets (p4kit.sh / the 1667 GT) use
# --gs 4 (the field on codes pooled 2x); here --gs 2 computes it on the codes' own 4.8 um grid, same radius (160 native voxels) and pair
# rule, so the rv side labels (the sampler's band + phase targets) come from the same raster. Restartable (each step skips done output,
# the phase resumes by shard).
#   p4A / p4B   Paris 4 June (wind-seed) / July (wind-assign to the June 19.2 um reference), segments fetched by
#               scripts/pod/fetchsegs.sh into $K/seg/PHercParis4; codes rastered at level 1 (raster --level 1).
#   1667A / 1667B  PHerc1667: the existing 4.798 um codes (/vesuvius/ufsm/gt/PHerc1667-wind-20261016/codes{A,B}).
# Env: K (work root, /mnt/scratch/univ/b1), U (binary with nn_band.o), GPU (CUDA device for the pair rule; -1 = CPU), LOCK (a flock file
# held around the phase step, e.g. /tmp/gpu1-bench.lock), T (threads), PS (shards at once), BOX (native box; default the label box),
# PHASE=0 (stop after the codes).
set -uo pipefail; SET=${1:?p4A|p4B|1667A|1667B}; K=${K:-/mnt/scratch/univ/b1}; U=${U:-$K/ufsm.labelfast}; GPU=${GPU:-1}; T=${T:-8}; PS=${PS:-3}
step() { echo "== $(date +%T) $*"; }
case $SET in
  p4*) SEG=$K/seg/PHercParis4; [ -d $SEG ] || SEG=$K/kit/seg/PHercParis4; G=$K/gt/paris4; AX=${AX:-/vesuvius/ufsm/teacher/axis-0.json}; SH=75784,32693,32693; UM=2.4
       DBOX=26624,4096,4096,43008,26624,26624; mkdir -p $G
       A=$(ls -d $SEG/20260623*-w*/mesh/*-on-20260411134726-2.4um.tifxyz | grep -v jordi); B=$(ls -d $SEG/20260701*/mesh/*-on-20260411134726-2.4um.tifxyz)
       step seed June; [ -s $G/qA/seed.jsonl ] || nice -n 5 $U wind-seed $G/qA $A --vol-um $UM --axis $AX > $G/seedA.log 2>&1 || { tail -3 $G/seedA.log; exit 1; }
       if [ $SET = p4A ]; then
         step raster June 4.8; [ -f $G/codesA48/done ] || { nice -n 5 $U raster $G/codesA48 --shape $SH --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qA/*.tifxyz > $G/rasterA48.log 2>&1 && touch $G/codesA48/done; tail -1 $G/rasterA48.log; }
         C=$G/codesA48; O=$G/phaseA48
       else
         step raster June 9.6 + ref 19.2; [ -f $G/codesA/zarr.json ] || nice -n 5 $U raster $G/codesA --shape $SH --um $UM --level 2 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qA/*.tifxyz 2>&1 | tail -1
         [ -d $G/ref ] || nice -n 5 $U downsample $G/codesA 9.6 $G/ref 19.2 --mode max --threads $T 2>&1 | tail -1
         step assign July; [ -s $G/qB/assign.jsonl ] || nice -n 5 $U wind-assign $G/ref 19.2 $G/qB $B --ref-um 19.2 --vol-um $UM --axis $AX --tol 2 --step 4 --threads $T > $G/assignB.log 2>&1 || { tail -3 $G/assignB.log; exit 1; }
         step raster July 4.8; [ -f $G/codesB48/done ] || { nice -n 5 $U raster $G/codesB48 --shape $SH --um $UM --level 1 --binary 0 --levels 1 --threads $T --shard 512 --value q.tif $G/qB/*.tifxyz > $G/rasterB48.log 2>&1 && touch $G/codesB48/done; tail -1 $G/rasterB48.log; }
         C=$G/codesB48; O=$G/phaseB48
       fi; CK=4.8;;
  1667*) G0=/vesuvius/ufsm/gt/PHerc1667-wind-20261016; G=$K/gt/1667; mkdir -p $G; AX=${AX:-/mnt/scratch/pods/PHerc1667/unroll/axis.json}; DBOX=""
       C=$G0/codes${SET#1667}; O=$G/phase${SET#1667}48; CK=4.798;;
esac
[ "${PHASE:-1}" = 0 ] && { step "codes only (PHASE=0)"; exit 0; }
B0=${BOX:-$DBOX}; L=""; [ -n "${LOCK:-}" ] && L="flock $LOCK"
# SHARDS=FILE: only these 9.6 um shards (z y x per line, 2048^3 native each; scripts/univ/b1_shardpack.py picks the packed ones), runs of
# consecutive x merged into one --box; otherwise one --box (BOX, default the label box) or the whole raster
if [ -n "${SHARDS:-}" ]; then BOXES=$(python3 -I -c "
import sys; r = sorted(tuple(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()); i = 0
while i < len(r):
    j = i
    while j + 1 < len(r) and r[j + 1][:2] == r[i][:2] and r[j + 1][2] == r[j][2] + 1: j += 1
    z, y, x = r[i]; print(f'{z*2048},{y*2048},{x*2048},2048,2048,{(j-i+1)*2048}'); i = j + 1" $SHARDS); else BOXES=${B0:-all}; fi
step phase $SET 4.8 "(GPU $GPU, $(echo "$BOXES" | wc -l) boxes)"; s=$(date +%s)
[ -e $O/done ] || { ok=1; for b in $BOXES; do bb=""; [ "$b" != all ] && bb="--box $b"
    CUDA_VISIBLE_DEVICES=$GPU $L nice -n 5 $U phase-raster $C $CK $O --gs 2 --pair 1 --radius 160 --axis $AX --threads $T --shards $PS --resume 1 \
      $bb $([ "$GPU" != -1 ] && echo --gpu 0 --gpu-slots 2) >> $O.log 2>&1 || { ok=0; echo "phase box $b failed"; }; done; [ $ok = 1 ] && touch $O/done; }
echo "phase $SET: $(( $(date +%s) - s )) s, $(find $O -type f -path '*/c/*' | wc -l) shards, $(du -sh $O | cut -f1); $(tail -1 $O.log)"

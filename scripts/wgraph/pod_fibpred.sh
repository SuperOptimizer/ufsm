#!/bin/bash
# scripts/wgraph/pod_fibpred.sh (perf pod): fibhv-d H / V fibre masks over a z slab of PHerc1667 20251217075048 at level 2 (9.6 um),
# for villa's vc_wgraph (1667 has no AFV and no published fibre prediction). Tube channels only (0 = H, 1 = V); traced on the desk.
# BOX = level-2 z,y,x,nz,ny,nx. Run inside: flock /root/gpu.lock timeout -k 10 ...
set -u
BOX=${BOX:-4096,0,0,512,3136,3232}
U=${U:-/root/master/build/ufsm}; CK=/root/wgraph/fibhv-d-30000.ckpt; CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp
KEY=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr; O=/root/wgraph/pred/z${BOX%%,*}; C=/root/wgraph/cache
mkdir -p $O $C
for ch in 0 1; do d=$O/H; [ $ch = 1 ] && d=$O/V
  [ -e $d/zarr.json ] && continue
  { time $U predict $CK $CT $KEY $d --um 2.399 --level 2 --box $BOX --channel $ch --gpu 0 --threads 8 --cache $C > $O/predict_c$ch.log 2>&1; } 2> $O/predict_c$ch.time \
    || { echo "predict c$ch failed"; tail -5 $O/predict_c$ch.log; }
done
touch $O/done

#!/bin/bash
# score.sh TAG CKPT: one latent-ink snapshot on the 1667 held-out GT, the same way the inkpod row was made (DESIGN 11-17): latink.sh
# (predict --heads ink,c12,c13 over the whole 1667 box at level 2, two tile grids blended, q 4) from the local CT, then the 17 labelled
# up/ wraps (w035-w051) rendered at scale 1 with the c12 / c13 papyrus gate -> /vesuvius/ufsm/inksmoke/eval/TAG/map/wNNN.tif.
# Desk card 1 under the shared locks (team order: suite excl, then opt); ITH 8 threads, lowprio.
set -u; TAG=${1:?TAG}; CK=${2:?CKPT}; U=${U:-/home/forrest/ufsm-inksdk/build/ufsm}
E=/vesuvius/ufsm/inksmoke/eval/$TAG; P=/vesuvius/ufsm/podun/PHerc1667; UP=$P/inksdk/up; CTR=/vesuvius/ufsm/v2/ct; CTK=PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr
[ -s $E/map/w051.tif ] && { echo "score $TAG: done before"; exit 0; }
mkdir -p $E/ink $E/in $E/map $E/r; cp $P/box.txt $P/axis.json $E/
LK="flock -s /tmp/ufsm-suite-locks/gpu1.excl flock -s /vesuvius/ufsm/opt/gpu1.lock"
[ -e $E/ink/ink.zarr/zarr.json ] || $LK env U=$U LATINK_CK=$CK ITH=8 UNROLL_CT="$CTR|$CTK|2.399|2" ~/bin/memcap -m 36G ~/bin/lowprio \
  bash /home/forrest/ufsm-inksdk/scripts/pod/latink.sh $E 1 $E/ink/ink.zarr > $E/latink.log 2>&1 || { echo "score $TAG: latink failed ($E/latink.log)"; exit 1; }
for n in $(seq 35 51); do ln -sfn $UP/w0$n $E/in/w0$n; done
for h in ink c12 c13; do v=$E/ink/$h.zarr; mkdir -p $E/r/$h; [ -e $E/r/$h/.done ] && continue
  $LK env CUDA_VISIBLE_DEVICES=1 UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 UFSM_RENDER_GPU_MEM=4 ~/bin/memcap -m 36G ~/bin/lowprio $U render $E/in $E/r/$h \
    --volume $v 9.596 --ct $CTR $CTK/2 --depth 1 --scale 1 --threads 8 > $E/render_$h.log 2>&1 && touch $E/r/$h/.done || { echo "score $TAG: render $h failed ($E/render_$h.log)"; exit 1; }
done
# (render --gate needs --fill: the gate as in strip_refill.sh, the ink render x max(c12, c13) / 255, img-gate)
for n in $(seq 35 51); do w=w0$n; [ -s $E/map/$w.tif ] && continue
  f() { ls $E/r/$1/$w*.tif 2>/dev/null | grep -v '_ct' | head -1; }; i=$(f ink); g1=$(f c12); g2=$(f c13)
  [ -n "$i" ] && [ -n "$g1" ] && [ -n "$g2" ] || { echo "score $TAG: $w: missing render"; exit 1; }
  $U img-gate $i $g1 $g2 --out $E/map/$w.tif > /dev/null 2>&1 || { echo "score $TAG: img-gate $w failed"; exit 1; }
done
echo "score $TAG: $(ls $E/map | wc -l) maps"

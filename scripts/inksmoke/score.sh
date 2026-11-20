#!/bin/bash
# score.sh TAG CKPT: one latent-ink snapshot on the 1667 held-out GT, the same way the inkpod row was made (DESIGN 11-17): latink.sh
# (predict --heads ink,c12,c13 over the whole 1667 box at level 2, two tile grids blended, q 4) from the local CT, then the 17 labelled
# up/ wraps (w035-w051) rendered at scale 1 with the c12 / c13 papyrus gate -> /vesuvius/ufsm/inksmoke/eval/TAG/map/wNNN.tif.
# DEPTH (4, 11-20): render --depth, the max over +-DEPTH voxels along the wrap normal. Depth 1 capped the 1667 human-label target at 0.80
# held-out AUC (the wraps sit 2-8 voxels from the label band at 9.6 um); depth 4 gives 0.886 (reader3 0.858). Scores before 11-20 used 1.
# Desk card 1 under the shared locks (team order: suite excl, then opt); ITH 8 threads, lowprio.
# Card 1 only (11-18): CUDA_VISIBLE_DEVICES=1 (PCI order, nvidia-smi's) and device 0 inside: latink's predict --gpu 1 alone opened card 0
# too (the desk's try9 card). A guard polls nvidia-smi every 2 s: any process of this job on card 0, or one above GPUCAP MiB (6000: the
# lead's card-1 limit beside try9's ~3.3 GB), is killed; the peak per process is logged to TAG/gpupeak.txt.
# FAST=1 (2026-11-20, fullvram; the default since the drivers skip their pairwrap after a FAST score (TAG/fast.done); FAST=0: the renders): latink over the held-out samples' 512 tiles only (LATINK_TILES, default
# FASTDIR/tiles-1667: ink-wrapscore --tiles-out, 131 + 125 tiles) and no render / img-gate / pairwrap2.py: `ufsm ink-wrapscore --vmap`
# samples ink x max(c12, c13) at each held-out sample's render pixel (render's rc_pixel at DEPTH, img-gate's table: the rendered maps'
# AUC exactly) and writes eval/pair_TAG.{heldout,hold,valonly}.tsv itself (pairwrap2.py's columns, vs reader3) from the cached sample
# table (FASTDIR/1667-up.uws). Any missing input, or > 0.1 % of the samples in ink shards the predict did not write, fails (exit 1, no
# tsv). WSU: a ufsm with ink-wrapscore and predict --tile-list (default /vesuvius/ufsm/bin/ufsm-fullvram): FAST predicts with it, not U.
# The blend writes only the tiles0 shards, level 0 (LATINK_BLEND_X). h1-3000 1 min 52 s, fr1-1000 (10.7 M) 3 min 14 s on the shared card 1.
set -u; TAG=${1:?TAG}; CK=${2:?CKPT}; U=${U:-/vesuvius/ufsm/bin/ufsm-inksdk}
E=/vesuvius/ufsm/inksmoke/eval/$TAG; P=/vesuvius/ufsm/podun/PHerc1667; UP=$P/inksdk/up; CTR=/vesuvius/ufsm/v2/ct; CTK=PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr
FAST=${FAST:-1}; FASTDIR=${FASTDIR:-/vesuvius/ufsm/inksmoke/fast}; WSU=${WSU:-/vesuvius/ufsm/bin/ufsm-fullvram}
[ $FAST = 1 ] && [ -e $E/fast.done ] && { echo "score $TAG: done before (FAST)"; exit 0; }
[ $FAST = 1 ] || { [ -s $E/map/w051.tif ] && { echo "score $TAG: done before"; exit 0; }; }
[ $FAST = 1 ] && { [ -x $WSU ] && [ -s $FASTDIR/1667-up.uws ] && [ -s $FASTDIR/tiles-1667/tiles0.txt ] || { echo "score $TAG: FAST needs $WSU, $FASTDIR/1667-up.uws, $FASTDIR/tiles-1667"; exit 1; }; }
exec 8> /vesuvius/ufsm/inksdk/score.lock; flock 8   # (one score at a time on the desk, team rule 11-20; the drivers queue here)
mkdir -p $E/ink $E/in $E/map $E/r; cp $P/box.txt $P/axis.json $E/
# GATEFROM=EVALDIR (a net without recto / verso rows, e.g. the full-res ink run): predict ink only; the papyrus gate (c12 / c13) is that
# eval's volumes (t9c-500: try9's rv on the same 1667 grid), rendered here at this DEPTH
[ -n "${GATEFROM:-}" ] && for h in c12 c13; do ln -sfn $GATEFROM/ink/$h.zarr $E/ink/$h.zarr; done
LK="flock -s /tmp/ufsm-suite-locks/gpu1.excl flock -s /vesuvius/ufsm/opt/gpu1.lock"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY   # (a card-0 MPS client env must not reach the card-1 predict)
export CUDA_DEVICE_ORDER=PCI_BUS_ID; GPUCAP=${GPUCAP:-6000}
C0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 0 {print $2}'); C1=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 1 {print $2}')
[ -n "$C0" ] && [ -n "$C1" ] || { echo "score $TAG: no card uuids"; exit 1; }
mine() { local p=$1; while [ "${p:-0}" -gt 1 ]; do [ $p = $$ ] && return 0; p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); done; return 1; }   # a descendant of this script
guard() { while :; do nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r g p m; do
    mine $p || continue; [ "$g" = "$C0" ] && { echo "$(date +%T) KILL pid $p on card 0" >> $E/gpupeak.txt; kill $p; continue; }
    echo "$(date +%T) $p $m" >> $E/gpumem.log; [ "$m" -gt $GPUCAP ] && { echo "$(date +%T) KILL pid $p at $m MiB > $GPUCAP" >> $E/gpupeak.txt; kill $p; }
  done; sleep 2; done; }
guard & GP=$!; trap 'kill $GP 2>/dev/null; [ -s $E/gpumem.log ] && sort -k3 -n $E/gpumem.log | tail -1 | sed "s/^/peak: /" >> $E/gpupeak.txt' EXIT
[ -e $E/ink/ink.zarr/zarr.json ] || $LK env CUDA_VISIBLE_DEVICES=1 U=$([ $FAST = 1 ] && echo $WSU || echo $U) LATINK_CK=$CK ITH=8 UNROLL_CT="$CTR|$CTK|2.399|2" ~/bin/memcap -m 36G ~/bin/lowprio \
  env LATINK_HEADS=${GATEFROM:+ink} LATINK_TILES=${LATINK_TILES:-$([ $FAST = 1 ] && echo $FASTDIR/tiles-1667)} LATINK_BLEND_X="${LATINK_BLEND_X:-$([ $FAST = 1 ] && echo --tile-list ${LATINK_TILES:-$FASTDIR/tiles-1667}/tiles0.txt --levels 1)}" bash ${LATINK:-$(dirname "$(readlink -f "$0")")/../pod/latink.sh} $E 0 $E/ink/ink.zarr > $E/latink.log 2>&1 || { echo "score $TAG: latink failed ($E/latink.log)"; exit 1; }
if [ $FAST = 1 ]; then
  ~/bin/memcap -m 12G ~/bin/lowprio $WSU ink-wrapscore $UP /vesuvius/ufsm/inkgt/segs-PHerc1667.json $(dirname $E)/pair_$TAG --ct-key $CTK --maps reader3=$P/ink/reader3/map \
    --vmap $TAG=$E/ink --depth ${DEPTH:-4} --ref reader3 --nonzero reader3 --split heldout,hold,valonly --need w035-w051 --table $FASTDIR/1667-up.uws --threads 6 > $E/wrapscore.log 2>&1 \
    || { echo "score $TAG: ink-wrapscore failed ($E/wrapscore.log)"; exit 1; }
  touch $E/fast.done; echo "score $TAG: FAST $(awk -F'\t' -v t=$TAG '$1 == t {print "heldout AUC", $3, "vs reader3", $4, "n", $2}' $(dirname $E)/pair_$TAG.heldout.tsv)"; exit 0
fi
for n in $(seq 35 51); do ln -sfn $UP/w0$n $E/in/w0$n; done
for h in ink c12 c13; do v=$E/ink/$h.zarr; mkdir -p $E/r/$h; [ -e $E/r/$h/.done ] && continue
  $LK env CUDA_VISIBLE_DEVICES=1 UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 UFSM_RENDER_GPU_MEM=4 ~/bin/memcap -m 36G ~/bin/lowprio $U render $E/in $E/r/$h \
    --volume $v 9.596 --ct $CTR $CTK/2 --depth ${DEPTH:-4} --scale 1 --threads 8 > $E/render_$h.log 2>&1 && grep -q "render: 17 wraps written" $E/render_$h.log && touch $E/r/$h/.done || { echo "score $TAG: render $h failed ($E/render_$h.log)"; exit 1; }
done
# (render --gate needs --fill: the gate as in strip_refill.sh, the ink render x max(c12, c13) / 255, img-gate)
for n in $(seq 35 51); do w=w0$n; [ -s $E/map/$w.tif ] && continue
  f() { ls $E/r/$1/$w*.tif 2>/dev/null | grep -v '_ct' | head -1; }; i=$(f ink); g1=$(f c12); g2=$(f c13)
  [ -n "$i" ] && [ -n "$g1" ] && [ -n "$g2" ] || { echo "score $TAG: $w: missing render"; exit 1; }
  $U img-gate $i $g1 $g2 --out $E/map/$w.tif > /dev/null 2>&1 || { echo "score $TAG: img-gate $w failed"; exit 1; }
done
echo "score $TAG: $(ls $E/map | wc -l) maps"

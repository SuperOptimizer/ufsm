#!/usr/bin/env bash
# scripts/univ/podkit_eval.sh HOST PORT [K]: the universal-net eval kit on a pod (2026-11-11), so U-step snapshot evals run off the desk with
# the DESK's wbench labels (pod-rebuilt codes differ by a turn offset and coverage, scripts/pod/DATAKIT.md). Copies only what eval.sh reads,
# in the layout of scripts/wbench/site-pod.env under K (default /workspace/ue):
#   K/wbgt/paris4-q9p6/q-9p6.zarr/9.6, K/gt/paris4/codesB/9.6, K/wbgt/PHerc1667-wind-20261016/codes/4.798: the shards of the 44 core boxes;
#   K/wbgt/PHerc{0139,0814,0500P2}-wind-*/codes: the 17 9.36 um GT boxes' shards; K/wb/xres: the cross-res labels (codes) + axes + boxes;
#   K/wbgt/paris4-wind-20261019/phaseA/9.6: the fibre HOLD box's shards (fiber-judge); K/small/axis + K/wbgt/axis: the umbilicus files;
#   K/wb/bin/ufsm (the pinned fitter b29b1bda: same fit config tag, so the ceil cache applies), K/wb/ceil/052747ed.tsv, K/wb/res (baseline rows);
#   K/wb/inkeval/crops; the ink3d-ext weights (INK=1). Label phase for the other boxes is not copied (11.5 GB): wbench's phase diagnostics
#   (gross / conf / cnt_*) come out na on the pod; ids / p05x / p1x / ceil are unaffected.
# Desk upload is ~5 MB/s: the kit is ~3.2 GB (+1.7 GB with INK=1). Reruns copy only what changed (rsync).
set -u
HOST=${1:?host}; PORT=${2:?port}; K=${3:-/workspace/ue}; INK=${INK:-1}
H=$(cd "$(dirname "$0")/../.." && pwd); S=$H/scripts/univ/shardsel.py; BT=$H/scripts/wbench/boxes.tsv; G=/vesuvius/ufsm/gt; W=/vesuvius/ufsm/wbench
E="ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o BatchMode=yes"; T=$(mktemp -d); trap 'rm -rf $T' EXIT
awk -F'\t' '/^#/ || ($10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/))' $BT > $T/core.tsv
put() {   # put SRC_ARRAY_DIR DST_ARRAY_DIR SCALE BOXES SCROLLS: the selected shards of one array (+ its parent group zarr.json)
  python3 -I $S $1 $3 $4 $5 0 > $T/list 2>> $T/sizes || return 1
  $E $HOST "mkdir -p $2" < /dev/null && rsync -a --files-from=$T/list -e "$E" $1/ $HOST:$2/ < /dev/null
  [ -e $(dirname $1)/zarr.json ] && rsync -a -e "$E" $(dirname $1)/zarr.json $HOST:$(dirname $2)/ < /dev/null; return 0; }
put $G/paris4-q9p6/q-9p6.zarr/9.6 $K/wbgt/paris4-q9p6/q-9p6.zarr/9.6 1 $T/core.tsv paris4
put $W/gtjuly/codesB/9.6 $K/gt/paris4/codesB/9.6 1 $T/core.tsv paris4j
put $G/PHerc1667-wind-20261016/codes/4.798 $K/wbgt/PHerc1667-wind-20261016/codes/4.798 2 $T/core.tsv 1667
put $G/PHerc0139-wind-20261016/codes/18.72 $K/wbgt/PHerc0139-wind-20261016/codes/18.72 0.5 $BT PHerc0139
put $G/PHerc0814-wind-20261019/codes/18.724 $K/wbgt/PHerc0814-wind-20261019/codes/18.724 0.5 $BT PHerc0814
put $G/PHerc0500P2-wind-20261019/codes/18.724 $K/wbgt/PHerc0500P2-wind-20261019/codes/18.724 0.5 $BT PHerc0500P2
printf '#\nHOLD\tparis4\t12288\t4096\t3584\t512\t576\t768\t2\tfibre hold\n' > $T/hold.tsv
put $G/paris4-wind-20261019/phaseA/9.6 $K/wbgt/paris4-wind-20261019/phaseA/9.6 1 $T/hold.tsv paris4   # (HOLD: 9.6 um voxels, as in boxes.tsv)
$E $HOST "mkdir -p $K/small/axis $K/wbgt/axis $K/wb/bin $K/wb/ceil $K/wb/res $K/wb/xres $K/wb/inkeval $K/ckpt" < /dev/null
rsync -a -e "$E" /vesuvius/ufsm/teacher/axis-0.json $HOST:$K/small/axis/paris4-axis-0.json < /dev/null
rsync -a -e "$E" /vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json $HOST:$K/small/axis/PHerc1667-umbilicus.json < /dev/null
rsync -a -e "$E" $G/axis/PHerc0139-umb-scan.json $G/axis/PHerc0814-20250804134230.json $G/axis/PHerc0500P2-20250820143440.json $HOST:$K/wbgt/axis/ < /dev/null
rsync -aL -e "$E" $W/bin/ufsm-b29b1bda $HOST:$K/wb/bin/ < /dev/null && $E $HOST "ln -sf ufsm-b29b1bda $K/wb/bin/ufsm" < /dev/null
rsync -a -e "$E" $W/ceil/052747ed.tsv $HOST:$K/wb/ceil/ < /dev/null
rsync -a -e "$E" $W/res/wind96*.tsv $W/res/lat-wz896-12000*.tsv $HOST:$K/wb/res/ < /dev/null
rsync -a --exclude 'phase' -e "$E" $W/xres/ $HOST:$K/wb/xres/ < /dev/null
rsync -a -e "$E" $W/inkeval/crops $HOST:$K/wb/inkeval/ < /dev/null
[ "$INK" = 1 ] && rsync -a -e "$E" /vesuvius/ufsm/teachers/dino3d/ckpt_78k_fullsup.pth $HOST:$K/ckpt/ < /dev/null
cat $T/sizes
echo "kit on $HOST:$K; total there: $($E $HOST "du -sh $K | cut -f1" < /dev/null)"

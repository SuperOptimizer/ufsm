#!/usr/bin/env bash
# scripts/pod/unroll.sh SCROLL: whole-scroll unroll on a rented pod within its detected limits. This script is a
# thin wrapper. scripts/unroll/unroll.sh (the e2e driver: prep fields ink ref slabs join wraps strip stripflat stripimg
# [perwrap faces]) does the work. It runs with the pod's local CT, checkpoints, box and axis from scripts/pod/setup.sh
# ($POD/scrolls/<S>.env), on one GPU, with threads and memory gates from the cgroup.
# The driver is resumable: state/<stage>.done markers, slab claims, predict --resume. Just re-run after any failure or reboot.
# Stop: touch $UR/<S>/STOP.
# Run it detached: setsid nohup bash scripts/pod/unroll.sh S > $POD/unroll-S.out 2>&1 < /dev/null &
#   Progress: $UR/<S>/progress.log; per-stage wall / CPU / max RSS: $UR/<S>/state/*.done.
# Pod defaults (per-stage RAM table in README "Pod runs"), all overridable by env:
#   TH = cgroup CPUs; PTH / ITH (predict threads) = TH; FIELDGPUS = FITGPUS = INKGPU = 0 (one GPU or MIG slice);
#   MINMEM 8 (cgroup-aware with `ufsm sysinfo --avail-gb`); CFMEM (count-field --mem-gb) = min(24, 40% of the cgroup RAM);
#   SNAP 1 (wrap-snap the wraps onto the sheets before the strip; SNAPT mix), PERWRAP 0. The strip is the deliverable. Per-wrap renders beside the strip would add 16-24 GB each.
#   CLEAN 1: fields and slab fits are deleted once the wraps exist.
set -u
S=${1:?usage: unroll.sh SCROLL}; shift || true
POD=${POD:-$([ "$(id -u)" = 0 ] && echo /root/pod || echo $HOME/pod)}
E=$POD/scrolls/$S.env; [ -e $E ] || { echo "pod/unroll: no $E: run scripts/pod/setup.sh $S first"; exit 2; }
. $E
REPO=${UNROLL_REPO:-$POD/ufsm}; D=$REPO/scripts/unroll
[ -e $D/unroll.sh ] || { echo "pod/unroll: $D/unroll.sh missing (ufsm master too old: re-run setup.sh)"; exit 2; }
NC=$($U sysinfo --ncpu 2>/dev/null || echo ${NCPU:-8}); LIM=$($U sysinfo --limit-gb 2>/dev/null || echo ${MEMGB:-60})
# the driver reads its scan row (CT root, key, um, level, axis, box, zr, strip args) from scrolls.tsv beside it: give it one with
# the pod's local CT (CTROOT / CTKEY env do the same in newer drivers)
B=$POD/drv; mkdir -p $B
cp $D/unroll.sh $B/ && cp $D/fields_cat.sh $B/ 2>/dev/null; for f in $D/*.sh; do [ -e $B/$(basename $f) ] || cp $f $B/; done
printf 'scroll\troot\tkey\tum\tlevel\taxis\tbox\tzr\tstrip\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $S $R $K $UM $LV $AXS $BOX "${ZR:--}" "${STRIPARGS:--}" > $B/scrolls.tsv
export U CK INKCK UNROLL_REPO=$REPO CTROOT=$R CTKEY=$K BOX
export UR=${UR:-$POD/unroll} TH=${TH:-$NC} PTH=${PTH:-$NC} ITH=${ITH:-$NC} FIELDGPUS=${FIELDGPUS:-0} FITGPUS=${FITGPUS:-0} INKGPU=${INKGPU:-0}
export MINMEM=${MINMEM:-8} CFMEM=${CFMEM:-$(( LIM * 2 / 5 < 24 ? LIM * 2 / 5 : 24 ))} PERWRAP=${PERWRAP:-0} CLEAN=${CLEAN:-1}
export RENDER_NT=${RENDER_NT:-$(( NC < 8 ? NC : 8 ))} UFSM_RENDER_MEM=${UFSM_RENDER_MEM:-$(( LIM / 4 < 16 ? LIM / 4 : 16 ))}
# renders keep a small device pool (4 GB): pods chain scrolls, so another scroll's slab fits can share the GPU with these renders
export UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4}
# a published ink (setup.sh fetched it): use it instead of predicting one
if [ -n "${INKPUB:-}" ] && [ -z "${INKCMD:-}" ]; then export INKCMD="$B/inkpub.sh"
  printf '#!/bin/sh\n# INKCMD O GPU OUT: the published ink (setup.sh) linked in; a missing one fails loudly (no dangling link)\n[ -e %s/zarr.json ] || { echo "inkpub: %s has no zarr.json" >&2; exit 1; }\nmkdir -p "$(dirname "$3")" && ln -sfn %s "$3"\n' "$INKPUB" "$INKPUB" "$INKPUB" > $B/inkpub.sh; chmod +x $B/inkpub.sh; fi
mkdir -p $UR
echo "$(date '+%F %T') pod/unroll $S: $NC CPUs, $LIM GB, TH $TH CFMEM $CFMEM MINMEM $MINMEM, CT $R/$K level $LV, box $BOX, axis $AXS, U $U" | tee -a $UR/pod-$S.log
$U sysinfo --disk $UR >> $UR/pod-$S.log 2>&1
exec bash $B/unroll.sh $S "$@"

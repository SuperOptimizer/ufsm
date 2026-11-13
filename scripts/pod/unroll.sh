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
#   slab-fit lanes on that GPU: FITLANES (auto in the driver: (device GB - 8) / 16, 1..3, threads = CPUs / lanes; 96 GB card: 3 lanes, 1.7x);
#   MINMEM 8 (cgroup-aware with `ufsm sysinfo --avail-gb`); CFMEM (count-field --mem-gb) = min(24, 40% of the cgroup RAM);
#   SNAP 1 (wrap-snap the wraps onto the sheets before the strip; SNAPT mix), PERWRAP 0. The strip is the deliverable. Per-wrap renders beside the strip would add 16-24 GB each.
#   CLEAN 1: fields and slab fits are deleted once the wraps exist.
set -u
S=${1:?usage: unroll.sh SCROLL}; shift || true
POD=${POD:-$([ "$(id -u)" = 0 ] && echo /root/pod || echo $HOME/pod)}
E=$POD/scrolls/$S.env; [ -e $E ] || { echo "pod/unroll: no $E: run scripts/pod/setup.sh $S first"; exit 2; }
. $E
# axis check (lead 11-12, after MANB: its corpus axis sat 3-10 mm off the papyrus): the corpus axis vs the papyrus centroid on level LV+3;
# off by > 2 mm at any of 5 z levels, or outside the papyrus -> the centroid axis (scrolls/S/axis_centroid.json). Skipped when the run already started (axis.json there).
if [ ! -e ${UR:-$POD/unroll}/$S/axis.json ] && [ "${AXCHECK:-1}" = 1 ]; then U=$U POD=$POD bash ${UNROLL_REPO:-$POD/ufsm}/scripts/pod/axcheck.sh $S $AXS; [ $? = 3 ] && AXS=$POD/scrolls/$S/axis_centroid.json && sed -i "s#AXS=[^;]*;#AXS=$AXS;#" $E; fi
REPO=${UNROLL_REPO:-$POD/ufsm}; D=$REPO/scripts/unroll
[ -e $D/unroll.sh ] || { echo "pod/unroll: $D/unroll.sh missing (ufsm master too old: re-run setup.sh)"; exit 2; }
NC=$($U sysinfo --ncpu 2>/dev/null || echo ${NCPU:-8}); LIM=$($U sysinfo --limit-gb 2>/dev/null || echo ${MEMGB:-60})
# the driver reads its scan row (CT root, key, um, level, axis, box, zr, strip args) from scrolls.tsv beside it: give it one with
# the pod's local CT (CTROOT / CTKEY env do the same in newer drivers)
B=$POD/drv/$S; mkdir -p $B   # per-scroll helper dir: concurrent queue lanes must not share scrolls.tsv or inkpub.sh
cp $D/unroll.sh $B/ && cp $D/fields_cat.sh $B/ 2>/dev/null; for f in $D/*.sh; do [ -e $B/$(basename $f) ] || cp $f $B/; done
printf 'scroll\troot\tkey\tum\tlevel\taxis\tbox\tzr\tstrip\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $S $R $K $UM $LV $AXS $BOX "${ZR:--}" "${STRIPARGS:--}" > $B/scrolls.tsv
export U CK INKCK UNROLL_REPO=$REPO CTROOT=$R CTKEY=$K BOX
export UR=${UR:-$POD/unroll} TH=${TH:-$NC} PTH=${PTH:-$NC} ITH=${ITH:-$NC} FIELDGPUS=${FIELDGPUS:-0} FITGPUS=${FITGPUS:-0} INKGPU=${INKGPU:-0}
export MINMEM=${MINMEM:-8} CFMEM=${CFMEM:-$(( LIM * 2 / 5 < 24 ? LIM * 2 / 5 : 24 ))} PERWRAP=${PERWRAP:-0} CLEAN=${CLEAN:-1}
export RENDER_NT=${RENDER_NT:-$(( NC < 8 ? NC : 8 ))} UFSM_RENDER_MEM=${UFSM_RENDER_MEM:-$(( LIM / 4 < 16 ? LIM / 4 : 16 ))}
# renders keep a small device pool (4 GB): pods chain scrolls, so another scroll's slab fits can share the GPU with these renders
export UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4}
export INKW=${INKW:-576}   # ink window: auto took 1600^3 / 90 GB on a free card and starved the other lane (0814, 11-12); 576 = 512 + 2 x halo 32 (feather needs window - 2 halo = k x shard)
# ink: the latent universal ink head (user 11-13: no more ift3), inkB recipe: scripts/pod/latink.sh (2-grid predict --heads ink,c12,c13 + ink-blend);
# the stripe renders gate the ink by max(p_recto, p_verso) (strip_refill.sh INKGATE)
export INKCMD=${INKCMD-${UNROLL_REPO:-$POD/ufsm}/scripts/pod/latink.sh} INKNAME=${INKNAME:-inkpod-9000} INKGATE="${UR:-$POD/unroll}/$S/ink/c12.zarr ${UR:-$POD/unroll}/$S/ink/c13.zarr"
# a published ink (setup.sh fetched it): use it instead of predicting one
if [ -n "${INKPUB:-}" ] && [ -z "${INKCMD:-}" ]; then export INKCMD="$B/inkpub.sh"
  printf '#!/bin/sh\n# INKCMD O GPU OUT: the published ink (setup.sh) linked in; a missing one fails loudly (no dangling link)\n[ -e %s/zarr.json ] || { echo "inkpub: %s has no zarr.json" >&2; exit 1; }\nmkdir -p "$(dirname "$3")" && ln -sfn %s "$3"\n' "$INKPUB" "$INKPUB" "$INKPUB" > $B/inkpub.sh; chmod +x $B/inkpub.sh; fi
mkdir -p $UR
echo "$(date '+%F %T') pod/unroll $S: $NC CPUs, $LIM GB, TH $TH CFMEM $CFMEM MINMEM $MINMEM, CT $R/$K level $LV, box $BOX, axis $AXS, U $U" | tee -a $UR/pod-$S.log
$U sysinfo --disk $UR >> $UR/pod-$S.log 2>&1
exec bash $B/unroll.sh $S "$@"

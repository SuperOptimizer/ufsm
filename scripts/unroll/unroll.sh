#!/usr/bin/env bash
# unroll.sh SCROLL: one whole-scroll unroll (the Paris 4 coarse-to-fine recipe) as ONE resumable run on one host (the desk: 2 GPUs, 32
# threads, 182 GB). Output root O=$UR/SCROLL (UR default /vesuvius/ufsm/unroll), names as in the Paris 4 run:
#   box.txt axis.json | fields/wind96 | ref-cf | slabs/sNNN + slabs.txt | joined | wraps/wNNN.sfc | ink/$INKNAME.zarr |
#   strip/{strip,img_fill,img_pap,fillmap,stretch.txt} (scripts/strip/strip_refill.sh layout) | per-wrap (PERWRAP=1, p4full layout): orig flat up r img prev |
#   faces/ (FACES=1) | state/<stage>.done (wall s, CPU s, max RSS) | logs/<stage>.log | progress.log
# Stages and their order (critical path first; -> = waits for):
#   prep    box (scrolls.tsv or the scanprep mask) + axis (scrolls.tsv file or ufsm axis)
#   fields  wind96 fit fields: the box split in z at a 1024-voxel boundary, one predict per GPU (same global tile grid), joined by
#           fields_cat.sh (shard files moved: the result is the single-GPU run's); FIELDGPUS="0" = one predict
#   ink     -> prep only; GPU INKGPU (1) from when the fields are done: INKCMD (the ink agent's recipe; default: the corpus 2-pass
#           ift3 ink, frozen GN, window 640 / 64 + a +256 offset pass, ink-blend) -> ink/$INKNAME.zarr
#   ctcal   -> fields (CTCAL=1 only, OFF by default): ufsm ct-cal --ct-um $LUM (the CT LEVEL's um) -> ct.cal; ref adds --ct-cal
#   ref     -> fields: count-field --model-only (REFCMD F REF [--ct-cal FILE] overrides; cf GPU, TH threads)
#   slabs   -> ref: snapped slab fits, FITLANES lanes per GPU in FITGPUS ("0 1"; auto: by device memory, 1..3); the INKGPU lanes start once the ink is done; then the
#           bad-slab check (fit.log: "illegal memory", "loss 0.00000", no final loss line): refit on 1 lane, up to 3 tries, else fail
#   join    -> slabs: fit-join --ref (refuses bad slabs) ; check -> join: wrap-check (wrapcheck.tsv verdicts; 0 CT peaks = fail)
#   wraps   -> check: ufsm wraps (.sfc, step 4, --zr from scrolls.tsv, --min-cm2, --col-r p90)
#   trim    -> wraps: the outer-end dup run (wrap-check) + near-empty outer wraps (on-papyrus <= PAPMIN 0.1; surf-stats) dropped: strip --last
#           = the first kept id (state/strip.args, the dropped ids in progress.log); TRIM=0 or a --last in STRIPARGS: off
#   snap    -> trim (SNAP=1 default; SNAP=0 off): ufsm wrap-snap wraps -> wraps_snap: each wrap moved along its rays onto the sheets
#           (smooth per-wrap offsets, order kept: never past half the gap to its neighbours, never into masked CT; SNAPT target, SNAPX
#           extra flags); the strip and its renders then use wraps_snap. CLEAN runs after it (the phase / mix targets read the fields)
#   strip   -> snap: ufsm strip ($STRIPARGS + --last) ; stripflat -> strip: strip-flatten (FLATX, default "--gpu 1"; FLATX= for the CPU) ;
#   stripimg -> stripflat + ink: stretch from the 3 middle stripes, filled stripe renders (FILLX) + strip-mosaic img_fill (rgb) and
#           img_pap (papyrus look) via scripts/strip/strip_refill.sh; renders UFSM_RENDER_GPU_MEM=4 (a fit may share the GPU), RGPU
#   perwrap -> wraps + ink (PERWRAP=1): flatten (GPU) + render + ink-compare per wrap, largest first (p4full3.sh recipe)
#   faces   -> perwrap (FACES=1): ufsm faces per flattened wrap
#   upload  never automatic: UPLOAD=1 and the scroll not prize-eligible (feedback: ask first); prints the command otherwise
# Heavy CPU stages wait for MemAvailable >= MINMEM (40) GB. STOP: touch $O/STOP (checked between stages / slabs / wraps). A stage
# whose done marker exists is skipped; partial stages resume (predict --resume parts, slab claims, per-wrap outputs).
# DRY=1: print the plan (commands) without running anything. Runs from a copy of these scripts in $O/bin (edits never touch a run).
# Env: U (ufsm), UR, TH (20), FIELDGPUS ("0 1"), FITGPUS ("0 1"), FITLANES (auto), INKGPU (1), INKCMD, REFCMD, CK, INKCK, INKNAME, MINMEM, PERWRAP,
#   CTROOT / CTKEY (replace the table's CT root / key), ZCACHE_GB (auto) / ZCACHE_ADMIT (0.6): the decoded-CT cache of the render stages,
#   TAG (output dir suffix), ITH / PTH (ink / fields predict threads), FACES, SPAR (4 per-wrap jobs), CLEAN (1: fields + slab fits deleted once the wraps exist), STRIPARGS / ZR / BOX (override scrolls.tsv).
set -u
S=${1:?usage: unroll.sh SCROLL}
H=$(cd "$(dirname "$0")" && pwd)
UR=${UR:-/vesuvius/ufsm/unroll}; O=$UR/$S${TAG:-}   # (TAG: e.g. a test crop beside the full run)
U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}
CK=${CK:-/vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt}; INKCK=${INKCK:-/vesuvius/ufsm/ink-ft/runs/ift3/snap/ift3-24000.ckpt}
# host limits (cgroup-aware on pods): ufsm sysinfo --ncpu / --avail-gb when the binary has it, else nproc / /proc/meminfo
ncpu() { local n; n=$($U sysinfo --ncpu 2> /dev/null) && [ "$n" -gt 0 ] 2> /dev/null && { echo $n; return; }; nproc; }
availgb() { local g; g=$($U sysinfo --avail-gb 2> /dev/null) && [ "$g" -ge 0 ] 2> /dev/null && { echo $g; return; }
            awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
INKNAME=${INKNAME:-ift3-24000}; TH=${TH:-$(ncpu)}; [ "$TH" -gt 20 ] 2> /dev/null && [ -z "${TH_ALL:-}" ] && TH=20; MINMEM=${MINMEM:-40}; INKGPU=${INKGPU:-1}
FIELDGPUS=${FIELDGPUS:-0 1}; FITGPUS=${FITGPUS:-0 1}; DRY=${DRY:-0}; I=64; OV=16
PRIZE="PHerc0125 PHerc0175A PHerc0175B PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0306B PHerc0358 PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0800 PHerc0813 PHerc0826 PHerc0846A PHerc0846B PHerc1203 PHerc1218 PHerc1545"

# the scan: scrolls.tsv (box / zr / strip args), else the corpus list.tsv (box from the mask)
row=$(awk -F'\t' -v s=$S '$1 == s' $H/scrolls.tsv)
REPO=${UNROLL_REPO:-$(cd $H/../.. && pwd)}
[ -n "$row" ] || row=$(awk -F'\t' -v s=$S '$1 == s { print $0 "\t-\t-\t-" }' $REPO/scripts/corpus/list.tsv)
[ -n "$row" ] || { echo "unroll: $S is in neither scrolls.tsv nor corpus/list.tsv"; exit 2; }
IFS=$'\t' read -r _ R K UM LV AXS TBOX TZR TSTRIP <<< "$row"
R=${CTROOT:-$R}; K=${CTKEY:-$K}   # (pods: the CT mirrored to local disk)
LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }')
BOX=${BOX:-$TBOX}; ZR=${ZR:-$TZR}; STRIPARGS=${STRIPARGS:-$TSTRIP}; [ "$STRIPARGS" = - ] && STRIPARGS=
F=$O/fields/wind96; REF=$O/ref-cf; INK=$O/ink/$INKNAME.zarr

# the run's own copy of the scripts (bash reads scripts incrementally: editing the repo must not change a running driver)
if [ "$DRY" != 1 ] && [ "${UNROLL_COPY:-0}" != 1 ]; then
  mkdir -p $O/bin && cp $H/unroll.sh $H/fields_cat.sh $H/scrolls.tsv $REPO/scripts/wbench/mkcrop.sh $REPO/scripts/strip/strip_run.sh $REPO/scripts/strip/strip_refill.sh $O/bin/ || exit 1
  UNROLL_COPY=1 UNROLL_REPO=$REPO exec bash $O/bin/unroll.sh "$@"
fi
MKCROP=$H/mkcrop.sh; [ -e $MKCROP ] || MKCROP=$REPO/scripts/wbench/mkcrop.sh
STRIPRUN=$H/strip_run.sh; [ -e $STRIPRUN ] || STRIPRUN=$REPO/scripts/strip/strip_run.sh
REFILL=$H/strip_refill.sh; [ -e $REFILL ] || REFILL=$REPO/scripts/strip/strip_refill.sh

[ "$DRY" = 1 ] || mkdir -p $O/state $O/logs
# the shared decoded-CT cache (src/zshm.c) for the render / faces / ink-compare stages: ZCACHE_GB (auto: 1/4 of the available RAM, at
# most 30 GB; off on the laptop = WSL), admit 0.6 (P4 w137 / w138 re-renders: 47-54% hits, a render 32.8 -> 20.0 s). The cache file
# is removed at the end of the run (also on STOP / failure). strip-mosaic: UFSM_JPEG_THREADS = cores / 2 (it encodes up to 2 stripes at once, --stripe-jobs 2).
# auto = OFF when the host has a CUDA device: the renders run on the GPU (GPU volcomp decode), and with the cache on they decode on the
# CPU instead (0490B stripe ws021: 12.0 s / 4.2 GB RSS without, 25.0 s / 29 GB with; same bytes; stripes are decoded about once anyway)
if [ "${ZCACHE_GB:-auto}" = auto ]; then
  if grep -qi microsoft /proc/version 2> /dev/null || $U sysinfo 2> /dev/null | grep -q '^gpu [0-9]'; then ZCACHE_GB=0; else ZCACHE_GB=$(( $(availgb) / 4 )); [ $ZCACHE_GB -gt 30 ] && ZCACHE_GB=30; fi
fi
ZCPATH=${UFSM_ZCACHE_PATH:-/dev/shm/ufsm-zcache-$(id -u)}
JPEGT=$(( $(ncpu) / 2 )); [ $JPEGT -lt 1 ] && JPEGT=1
zcon() { export UFSM_ZCACHE_GB=$ZCACHE_GB UFSM_ZCACHE_ADMIT=${ZCACHE_ADMIT:-0.6}; }   # (called inside the stages' subshells)
zcclean() { [ "$DRY" = 1 ] || [ "$ZCACHE_GB" = 0 ] || rm -f "$ZCPATH"; }
[ "$DRY" = 1 ] || trap zcclean EXIT
log() { if [ "$DRY" = 1 ]; then echo "    log: $*"; else echo "$(date '+%F %T') $*" | tee -a $O/progress.log; fi; }
stopped() { [ -e $O/STOP ]; }
done_() { [ -e $O/state/$1.done ]; }
memok() { [ "$DRY" = 1 ] && return 0; while [ $(availgb) -lt $MINMEM ]; do sleep 30; done; }
# X LOG CMD..: a command, nice 10, timed (/usr/bin/time -v appended to LOG); DRY: printed
X() { local l=$1; shift; if [ "$DRY" = 1 ]; then echo "    $*  > $l"; return 0; fi
      /usr/bin/time -v nice -n 10 "$@" >> $l 2>&1; }
# stage NAME FN: FN unless done; the marker records wall s, then the CPU s and max RSS summed / maxed over the stage's timed commands
stage() {
  local st=$1 fn=$2 t0 rc l=$O/logs/$1.log
  done_ $st && return 0; stopped && return 1
  if [ "$DRY" = 1 ]; then echo "[$st]"; $fn; return 0; fi
  log "$st start"; t0=$(date +%s); : >> $l; local m0=$(wc -l < $l)
  $fn; rc=$?
  local cpu rss; read -r cpu rss < <(tail -n +$((m0 + 1)) $l | awk -F': ' '/User time|System time/ {c += $2} /Maximum resident/ {if ($2 > r) r = $2} END {printf "%.0f %.1f\n", c, r / 1048576}')
  if [ $rc = 0 ]; then echo "$(( $(date +%s) - t0 )) s wall, ${cpu:-0} CPU s, ${rss:-0} GB max RSS" > $O/state/$st.done; log "$st done: $(cat $O/state/$st.done)"
  else log "$st FAILED rc $rc (logs/$st.log)"; fi
  return $rc
}
waitfor() { [ "$DRY" = 1 ] && { echo "    (waits for state/$1.done)"; return 0; }; while ! done_ $1; do stopped && return 1; [ -e $O/state/$1.failed ] && return 1; sleep 30; done; }

prep() {
  if [ -n "$BOX" ] && [ "$BOX" != - ]; then [ "$DRY" = 1 ] && echo "    box $BOX" || echo $BOX > $O/box.txt
  else
    local id=$(basename $K | cut -c1-14) m sh GZ GY GX cell
    m=/vesuvius/ufsm/u0/prep/$S-$id.mask.raw; sh=$(jq -r '.shape | join(" ")' $R/$K/$LV/zarr.json)
    read -r GZ GY GX < <(jq -r '.grid | join(" ")' $m.json); cell=$(jq -r .cell $m.json)
    [ "$DRY" = 1 ] && { echo "    box from $m (corpus desk.sh prep)"; } || od -An -v -tu1 -w$((GY * GX)) $m | awk -v X=$GX -v c=$cell -v sh="$sh" '
      { for (i = 1; i <= NF; i++) if ($i > 0) { y = int((i - 1) / X); x = (i - 1) % X; z = NR - 1
          if (!n++) { z0 = z1 = z; y0 = y1 = y; x0 = x1 = x }
          if (z < z0) z0 = z; if (z > z1) z1 = z; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x } }
      END { split(sh, s, " "); a[1] = z0; a[2] = y0; a[3] = x0; b[1] = z1; b[2] = y1; b[3] = x1
        for (d = 1; d <= 3; d++) { lo = (a[d] - 1) * c; if (lo < 0) lo = 0; lo = int(lo / 512) * 512; hi = (b[d] + 2) * c; if (hi > s[d]) hi = s[d]; o[d] = lo; e[d] = hi - lo }
        printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], e[1], e[2], e[3] }' > $O/box.txt
  fi
  if [ "$AXS" = auto ]; then X $O/logs/prep.log $U axis $R $K $O/axis.json || return 1
  else [ "$DRY" = 1 ] && echo "    axis $AXS" || cp $AXS $O/axis.json; fi
  [ "$DRY" = 1 ] || [ -s $O/box.txt ]
}

# --- fields: one predict per GPU on z parts (whole 1024-voxel multiples; the last part takes the rest) ---
occlv() {   # --occupancy L for the fields: the first of levels LV+3, LV+2, LV+4 the CT has (local root or mirror URL); none: no flag
  local l; for l in $((LV + 3)) $((LV + 2)) $((LV + 4)); do
    case $R in http://*|https://*) curl -sfI $R/$K/$l/zarr.json > /dev/null && { echo "--occupancy $l"; return; };;
                     *) [ -e $R/$K/$l/zarr.json ] && { echo "--occupancy $l"; return; };; esac
  done
}
fpred() {   # fpred GPU BOX OUT
  X $O/logs/fields.log $U predict $CK $R $K $3.tmp --um $UM --level $LV --box $2 --window-z 384 --halo-z 64 --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema 1 --q 8 --threads ${PTH:-14} --gpu $1 --axis $O/axis.json --levels 1 --fit-zarr $3 --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 $(occlv) &&
    { [ "$DRY" = 1 ] || { rm -rf $3.tmp; [ -e $3/fields.json ]; }; }
}
fields() {
  local box; box=$([ "$DRY" = 1 ] && echo ${BOX:-BOX} || cat $O/box.txt); local g=($FIELDGPUS) n=${#g[@]}
  [ "$DRY" = 1 ] || { rm -rf $F; mkdir -p $O/fields; }
  local z0 dz y0 x0 dy dx k a b pids=() parts=(); IFS=, read -r z0 y0 x0 dz dy dx <<< "$box"
  [ $(( (dz / n) / 1024 )) -lt 1 ] && n=1   # (parts of >= 1024 voxels each)
  if [ $n = 1 ]; then fpred ${g[0]} $box $F; return; fi
  # z parts: equal shares rounded to 1024 voxels (= 512 field cells at h 2, a multiple of the 128-cell field shard and of the 256 z stride)
  for k in $(seq 0 $((n - 1))); do
    a=$(( z0 + (dz * k / n) / 1024 * 1024 )); b=$(( k == n - 1 ? z0 + dz : z0 + (dz * (k + 1) / n) / 1024 * 1024 ))
    parts+=($F.p$k); [ "$DRY" = 1 ] || rm -rf $F.p$k $F.p$k.tmp
    fpred ${g[$k]} $a,$y0,$x0,$((b - a)),$dy,$dx $F.p$k & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; [ $rc = 0 ] || return 1
  local acc=${parts[0]} j; for j in $(seq 1 $((n - 1))); do
    if [ "$DRY" = 1 ]; then echo "    fields_cat.sh $F.c$j $acc ${parts[$j]}"; acc=$F.c$j; continue; fi
    bash $H/fields_cat.sh $F.c$j $acc ${parts[$j]} >> $O/logs/fields.log 2>&1 || return 1; acc=$F.c$j; done
  [ "$DRY" = 1 ] && echo "    mv $acc $F" || mv $acc $F
}

# --- ctcal (CTCAL=1 only; OFF by default, the user has not decided): the density turn count calibrated on the CT's papyrus peaks ---
# --ct-um is the um of the CT LEVEL read ($LUM = UM x 2^LV, e.g. 9.596 at level 2 of a 2.399 um scan; a pod passed 2.399 and the
# calibration read 0 peaks): the overall CT peaks per density turn must lie in CTCAL_RANGE (0.4,2.5) or the stage fails
ctcal() {
  X $O/logs/ctcal.log $U ct-cal $F $O/ct.cal.tmp --ct $R $K/$LV --ct-um $LUM --threads $TH || return 1
  [ "$DRY" = 1 ] && { echo "    (fails unless the overall CT peaks per density turn is inside ${CTCAL_RANGE:-0.4,2.5})"; return 0; }
  local k; k=$(grep -oE 'CT peaks per density turn [0-9.]+ overall' $O/logs/ctcal.log | tail -1 | awk '{print $6}')
  if [ -z "$k" ] || ! awk -v k=$k -v r=${CTCAL_RANGE:-0.4,2.5} 'BEGIN { split(r, a, ","); exit !(k >= a[1] && k <= a[2]) }'; then
    log "ctcal: CT peaks per density turn '${k:-none}' outside ${CTCAL_RANGE:-0.4,2.5}: wrong --ct-um ($LUM) or CT level ($K/$LV)?"; return 1; fi
  mv $O/ct.cal.tmp $O/ct.cal; log "ctcal: CT peaks per density turn $k (--ct-um $LUM, level $LV)"
}
ref() {
  [ "$DRY" = 1 ] || rm -rf $REF
  local cal=; [ "${CTCAL:-0}" = 1 ] && cal="--ct-cal $O/ct.cal"
  if [ -n "${REFCMD:-}" ]; then X $O/logs/ref.log $REFCMD $F $REF $cal; else
  X $O/logs/ref.log $U count-field $F $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $REF --model-only 1 --model-stride ${MSTRIDE:-4} \
    --threads $TH --mem-gb ${CFMEM:-24} $cal; fi && { [ "$DRY" = 1 ] || [ -e $REF/model.json ]; }
}

# --- ink (another agent's recipe via INKCMD SCROLLDIR GPU OUT; default the corpus 2-pass) ---
ink() {
  [ "$DRY" = 1 ] || mkdir -p $O/ink
  # INKCMD set but empty (a pod env with an unset path, "INKCMD=$X"): fail loudly instead of falling back to the 2-pass prediction;
  # any INKCMD must leave a readable zarr at $INK (a dangling link from a missing published ink fails here, not in the renders)
  if [ -n "${INKCMD+x}" ] && [ -z "${INKCMD// /}" ]; then log "ink: INKCMD is set but EMPTY: refusing (unset it for the default 2-pass ink)"; return 1; fi
  if [ -n "${INKCMD:-}" ]; then X $O/logs/ink.log $INKCMD $O $INKGPU $INK || { log "ink: INKCMD '$INKCMD' failed (logs/ink.log)"; return 1; }
    [ "$DRY" = 1 ] || [ -e $INK/zarr.json ] || { log "ink: INKCMD '$INKCMD' left no zarr at $INK (zarr.json missing; dangling link?)"; return 1; }
    return 0; fi
  local box cal bb bo; box=$([ "$DRY" = 1 ] && echo ${BOX:-BOX} || cat $O/box.txt)
  cal=$(echo $box | awk -F, '{ zc = $1 + int($4 / 2); z0 = zc - 1024; if (z0 < $1) z0 = $1; nz = 2048; if (nz > $4) nz = $4; printf "%d,%d,%d,%d,%d,%d", z0, $2, $3, nz, $5, $6 }')
  bb=$(echo $box | awk -F, '{ printf "%d,%d,%d,%d,%d,%d", $1 + 256, $2 + 256, $3 + 256, $4 - 256, $5 - 256, $6 - 256 }'); bo=$(echo $bb | cut -d, -f1-3)
  local P="--um $UM --level $LV --shard 512 --window 640 --halo 64 --ema 1 --q 32 --threads ${ITH:-12} --gpu $INKGPU $(occlv)"
  [ -s $O/ink/gn.bin ] || { X $O/logs/ink.log $U predict $INKCK $R $K $O/ink/gncal.zarr $P --box $cal --full-grid 1 --levels 1 --no-out 1 --gn-save $O/ink/gn.tmp &&
                             { [ "$DRY" = 1 ] || { mv $O/ink/gn.tmp $O/ink/gn.bin; rm -rf $O/ink/gncal.zarr; }; } || return 1; }
  X $O/logs/ink.log $U predict $INKCK $R $K $O/ink/a.zarr $P --box $box --full-grid 1 --resume 1 --levels 4 --gn-frozen $O/ink/gn.bin || return 1
  X $O/logs/ink.log $U predict $INKCK $R $K $O/ink/b.zarr $P --box $bb --resume 1 --levels 1 --gn-frozen $O/ink/gn.bin || return 1
  [ "$DRY" = 1 ] || rm -rf $INK
  X $O/logs/ink.log $U ink-blend $INK --um $LUM --tile 512,512,512 --grids 0,0,0:256,256,256 --origins 0,0,0:$bo --levels 4 --shard 512 --q 32 \
    --threads ${ITH:-12} $O/ink/a.zarr $O/ink/b.zarr || return 1
  [ "$DRY" = 1 ] || rm -rf $O/ink/a.zarr $O/ink/b.zarr
}

# --- slab fits: one lane per GPU, claims by flock (fd.sh lane) ---
lane() {   # lane GPU
  local g=$1 k s d c0 c1 a b t0 rc ze ny nx dir
  if [ "$DRY" = 1 ]; then echo "    lane GPU $g: per slab k: mkcrop.sh $F slabs/sNNN/f (64 cells + 16 overlap); $U fit slabs/sNNN/f --gpu $g --out slabs/sNNN/fit --dir REFDIR --ref-model $REF --ref-snap 1 --uw-repair 0 [phase gate on by default; FITX=\"--phase-gate 0\" off] ${FITX:-}"; return 0; fi
  ze=$(jq -r '.shape[0]' $F/fields.json); ny=$(jq -r '.shape[1]' $F/fields.json); nx=$(jq -r '.shape[2]' $F/fields.json); dir=$(jq -r .dir $REF/model.json)
  for k in $(seq 0 $(( (ze + I - 1) / I - 1 ))); do
    stopped && return 0
    s=$(printf 's%03d' $k); d=$O/slabs/$s; mkdir -p $d
    exec 9> $d/claim; flock -n 9 || continue; [ -e $d/done ] && continue
    c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
    rm -rf $d/f $d/fit; bash $MKCROP $F $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null || continue
    echo "$c0 $c1 $a $b" > $d/cells; memok; t0=$(date +%s)
    # FIT_MANAGED=1 (the last refit try of a failed slab) puts the fields in managed memory (fit picks it itself when they leave too little
    # device headroom since 6126e23; this is the fallback)
    local man=${FIT_MANAGED:-0}
    env UFSM_FIT_THREADS=$FTH_LANE $([ "$man" = 1 ] && echo UFSM_FIT_MANAGED=1) /usr/bin/time -v nice -n 5 $U fit $d/f --gpu $g --out $d/fit --dir $dir --ref-model $REF --ref-snap 1 --uw-repair 0 ${FITX:-} > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s$([ "$man" = 1 ] && echo " managed")" >> $O/slabs.txt
    if [ $rc = 0 ]; then touch $d/done; elif grep -q "no data for the prior" $d/fit.log; then touch $d/empty $d/done; fi
    rm -rf $d/f; exec 9>&-
  done
}
# fitlanes: the slab-fit lanes, a GPU id per lane. Each GPU of FITGPUS gets FITLANES lanes; FITLANES unset means auto:
# (device GB - 8) / 16, clamped to 1..3, so 16-24 GB cards get 1 lane, 48 GB MIG slices 2, and 96 GB cards 3.
# Measured on a 96 GB RTX PRO 6000 with 6 Paris 3 slabs and threads = CPUs / lanes: 1 lane 82.8 s, 2 lanes 54.8 s,
# 3 lanes 47.9 s, delta.f32 byte-identical. A fit takes ~10-16 GB of device memory. The device size comes from
# `ufsm sysinfo` (CUDA order, MIG-aware); when it is unknown, 1 lane.
fitlanes() {
  local g n gb i out=
  for g in $FITGPUS; do
    n=${FITLANES:-}
    if [ -z "$n" ]; then gb=$($U sysinfo 2> /dev/null | awk -v g=$g '$1 == "gpu" && $2 == g ":" && match($0, /, [0-9.]+ GB \(/) { print int(substr($0, RSTART + 2, RLENGTH - 6)) }')
      n=$(( (${gb:-0} - 8) / 16 )); [ $n -lt 1 ] && n=1; [ $n -gt 3 ] && n=3; fi
    for i in $(seq $n); do out+="$g "; done
  done; echo $out
}
slabs() {
  local g pids=() ng lanes
  # fit threads per lane: the host's CPUs over the lanes (fits are thread-count independent: 0490B s020/s021 models equal at 10 / 20;
  # 1 lane 10 -> 20 threads: 8 slabs 163 -> 150 s; 2 lanes per 16 GB GPU with the GPU lock: 156 s, so 1 lane there; 96 GB: 3, fitlanes)
  lanes=$(fitlanes); [ "$DRY" = 1 ] && echo "    slab lanes (GPU per lane): $lanes"
  ng=$(echo $lanes | wc -w); FTH_LANE=${FTHREADS:-$(( $(ncpu) / (ng > 0 ? ng : 1) ))}; [ $FTH_LANE -lt 2 ] && FTH_LANE=2
  for g in $lanes; do
    if [ "$g" = "$INKGPU" ] && ! done_ ink; then { [ "$DRY" = 1 ] && echo "    (GPU $g lane starts once the ink is done or failed)" || while ! done_ ink && ! stopped && [ ! -e $O/state/ink.failed ]; do sleep 30; done; lane $g; } &
    else lane $g & fi; pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  [ "$DRY" = 1 ] && return 0
  local ns=$(( ($(jq -r '.shape[0]' $F/fields.json) + I - 1) / I )) nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l)
  cat $O/slabs/s???/fit.log | grep -E 'User time|System time|Maximum resident' >> $O/logs/slabs.log   # (the stage marker's CPU s)
  log "slabs $nd of $ns done"
  # failed (rc != 0) and bad slabs (a fit that "succeeded" with an illegal memory access, a zero loss or no final loss line: pods 10-31, GPU
  # memory pressure): refit on ONE lane (the first GPU of FITGPUS), up to 3 tries, the last with the fields in managed memory; the stage
  # fails if any stays failed / bad (the join refuses bad ones too)
  local try bad; for try in 1 2 3; do
    bad=$(badslabs all); [ -z "$bad" ] && break
    log "slabs: failed / bad fits (try $try): $(for d in $bad; do basename $d; done | tr '\n' ' ')-> refit on 1 lane$([ $try = 3 ] && echo ', managed memory')"
    for d in $bad; do [ -e $d/fit.log ] && mv -f $d/fit.log $d/fit.bad$try.log; rm -rf $d/done $d/fit $d/empty; done
    FIT_MANAGED=$([ $try = 3 ] && echo 1 || echo 0) lane ${FITGPUS%% *}
  done
  bad=$(badslabs all); [ -z "$bad" ] || { log "slabs: failed / bad fits remain after 3 refits: $(for d in $bad; do basename $d; done | tr '\n' ' ')"; return 1; }
  nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l); log "slabs $nd of $ns done after the checks"; [ $nd = $ns ]
}
# badslabs [all]: the slab dirs whose (non-empty) fit is bad: fit.log has "illegal memory" or "loss 0.00000", or lacks the final "fit: ...
# loss X" line; all: also the slabs that are not done (a failed fit)
badslabs() {
  local d ns=$(( ($(jq -r '.shape[0]' $F/fields.json) + I - 1) / I )) k; for k in $(seq 0 $((ns - 1))); do d=$O/slabs/$(printf 's%03d' $k)
    if [ ! -e $d/done ]; then [ "${1:-}" = all ] && echo $d; continue; fi; [ -e $d/empty ] && continue
    if [ ! -e $d/fit.log ] || grep -qE 'illegal memory|loss 0\.00000' $d/fit.log || ! grep -qE '^fit: .*loss [0-9]' $d/fit.log; then echo $d; fi
  done
}
join() {
  local args=() d c0 c1 a b bad
  if [ "$DRY" = 1 ]; then X $O/logs/join.log $U fit-join $O/joined $F --blend 16 --ref $REF "slabs/sNNN/fit 2c0 2c1 ..."; return; fi
  bad=$(badslabs); [ -z "$bad" ] || { log "join: REFUSED: bad slab fits $(for d in $bad; do basename $d; done | tr '\n' ' ')(rm state/slabs.done to refit)"; return 1; }
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; [ -e $d/empty ] && continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; X $O/logs/join.log $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}"
}
# check: wrap-check of the joined model (CT papyrus peaks between adjacent wraps along radial rays; per-wrap verdicts real / dup / merged
# in wrapcheck.tsv). --ct-um is the CT level's um ($LUM); 0 CT peaks per turn overall = a wrong level / um: the stage fails
check() {
  X $O/logs/check.log $U wrap-check $O/joined $F --ct $R $K/$LV --ct-um $LUM --tsv $O/wrapcheck.tsv.tmp --threads $TH || return 1
  [ "$DRY" = 1 ] && return 0
  [ -s $O/wrapcheck.tsv.tmp ] && [ $(awk -F'\t' 'NR > 1 && $NF != ""' $O/wrapcheck.tsv.tmp | wc -l) -gt 0 ] || { log "check: wrap-check wrote no verdicts"; return 1; }
  awk -F'\t' 'NR > 1 { n++; e += $3 } END { exit !(n > 0 && e / n < 0.999) }' $O/wrapcheck.tsv.tmp ||
    { log "check: wrap-check found no CT peaks (empty share ~1 on every wrap): wrong --ct-um $LUM or CT level $K/$LV?"; return 1; }
  mv $O/wrapcheck.tsv.tmp $O/wrapcheck.tsv
  log "check: $(grep -E '^wrap-check: verdicts' $O/logs/check.log | tail -1)"
}
wraps() {
  local zr=; [ -n "$ZR" ] && [ "$ZR" != - ] && zr="--zr $ZR"
  [ "$DRY" = 1 ] || rm -rf $O/wraps
  X $O/logs/wraps.log $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH $zr --min-cm2 ${MINCM2W:-0.5}
}
# trim: the strip's outer end. Wrap ids increase OUTWARD and the strip walks outermost first (column 0 = the highest id), so the outer
# end is the highest ids. (1) From the highest wrap-check verdict id walk down while the verdict is dup (the ref's outer over-count:
# duplicate turns). (2) With no dup run, start from the highest wrap written instead. (3) Keep walking down while the wrap is near-empty:
# on-papyrus (surf-stats --ct-only, CT >= 80 within +-15 um) <= PAPMIN (0.1). The first kept id becomes strip --last (state/strip.args);
# the dropped ids are logged. STRIPARGS with its own --last, or TRIM=0, skips this. CLEAN=1 deletes the field planes + slab fits after the snap stage.
onpap() {   # onpap ID..: "id share" per wrap (surf-stats --per-wrap --ct-only on a link dir of those wraps)
  local d=$O/trim.tmp i; rm -rf $d; mkdir -p $d; for i in "$@"; do ln -s $O/wraps/$(printf 'w%03d' $i).sfc $d/; done
  nice -n 10 $U surf-stats $d --fields $F --ct $R $K/$LV --ct-um $LUM --ct-only 1 --per-wrap --threads $TH 2>> $O/logs/trim.log |
    tee -a $O/logs/trim.log | awk '$1 ~ /^w[0-9]+$/ && NF >= 8 { print substr($1, 2) + 0, $NF }'
  rm -rf $d
}
trim() {
  [ "$DRY" = 1 ] && { echo "    strip --last = first kept id walking down from the top: wrapcheck.tsv dup run, then wraps with on-papyrus <= ${PAPMIN:-0.1} (surf-stats --ct-only)"; return 0; }
  local args=$STRIPARGS ids top last k i v p dropped= batch=() pap=$O/trim.pap; : > $O/state/strip.args
  if [ "${TRIM:-1}" != 1 ] || [[ " $STRIPARGS " == *" --last "* ]]; then log "trim: skipped (TRIM=${TRIM:-1}, STRIPARGS '$STRIPARGS')"
  else
    ids=($(ls $O/wraps | sed -n 's/^w\([0-9]*\)\.sfc$/\1/p' | awk '{print $1 + 0}' | sort -rn)); [ ${#ids[@]} -gt 0 ] || { log "trim: no wraps"; return 1; }
    declare -A V; while IFS=$'\t' read -r w _ _ _ _ _ _ _ _ _ _ _ _ v; do [[ $w =~ ^w[0-9]+$ ]] && V[$((10#${w#w}))]=$v; done < $O/wrapcheck.tsv
    top=$(printf '%s\n' "${!V[@]}" | sort -rn | head -1); k=0
    # (1) the dup run below the highest verdict id; (2) no dup run: from the highest wrap written
    if [ -n "$top" ] && [ "${V[$top]}" = dup ]; then
      while [ $k -lt ${#ids[@]} ] && { [ ${ids[$k]} -gt $top ] || [ "${V[${ids[$k]}]:-}" = dup ]; }; do dropped+=" ${ids[$k]}"; k=$((k + 1)); done
      log "trim: outer-end dup run (wrap-check verdicts) + every wrap above it: $(echo $dropped | wc -w) wraps"
    fi
    # (3) the near-empty outer tail: on-papyrus <= PAPMIN, computed from the top in batches of 1, 4, then 16 wraps (usually the first
    # wrap below the dup run decides: one surf-stats of one wrap, ~5 s instead of ~65 s for 16 outer wraps on 0490B)
    declare -A P; : > $pap; local nb=1
    while [ $k -lt ${#ids[@]} ]; do
      i=${ids[$k]}
      if [ -z "${P[$i]:-}" ]; then batch=("${ids[@]:$k:$nb}"); nb=$(( nb < 16 ? nb * 4 : 16 )); while read -r a b; do P[$a]=$b; echo "$a $b" >> $pap; done < <(onpap "${batch[@]}"); [ -n "${P[$i]:-}" ] || P[$i]=0; fi
      awk -v p=${P[$i]} -v m=${PAPMIN:-0.1} 'BEGIN { exit !(p <= m) }' || break
      dropped+=" $i"; k=$((k + 1))
    done
    [ $k -lt ${#ids[@]} ] || { log "trim: every wrap would be dropped: refusing (TRIM=0 to skip)"; return 1; }
    last=${ids[$k]}
    if [ -n "$dropped" ]; then args="$STRIPARGS --last $last"
      log "trim: strip --last $last (first kept: verdict ${V[$last]:-none}, on-papyrus ${P[$last]:-?}); dropped $(echo $dropped | wc -w) outer wraps:$(for i in $dropped; do printf ' w%03d(%s,%s)' $i "${V[$i]:--}" "${P[$i]:--}"; done)"
    else log "trim: nothing dropped (top w$(printf %03d ${ids[0]}): verdict ${V[${ids[0]}]:-none}, on-papyrus ${P[${ids[0]}]:-?})"; fi
  fi
  echo "$args" > $O/state/strip.args
  return 0
}
# snap: the wraps onto the sheets (src/wrapsnap.c): along their rays (z and the ray angle kept: the strip's columns / rows unchanged), each
# within (1/2 - margin) of the radial gap to its neighbouring wraps (order kept, two wraps never on one sheet), a smooth offset field per
# wrap fitted to the target's ridges (SNAPT: ct = CT ridges, phase / recto = the fields' sheet centres / recto, mix = CT ridges weighted
# by the network phase), never into masked CT. Then CLEAN=1 deletes the field planes + slab fits (the joined model is standalone).
wrapsdir() { if [ "${SNAP:-1}" = 1 ] && { [ "$DRY" = 1 ] || [ -e $O/state/snap.done ]; }; then echo $O/wraps_snap; else echo $O/wraps; fi; }
snap() {
  if [ "${SNAP:-1}" = 1 ]; then
    local t=${SNAPT:-mix}
    [ "$DRY" = 1 ] || rm -rf $O/wraps_snap
    # (a run whose fields were already cleaned: the CT target)
    [ "$DRY" != 1 ] && [ "$t" != ct ] && [ ! -e $F/pc/zarr.json ] && { log "snap: no field planes in $F (cleaned): --target ct"; t=ct; }
    X $O/logs/snap.log $U wrap-snap $O/wraps $O/wraps_snap --target $t --axis $O/axis.json --axis-um $UM --ct $R $K/$LV --ct-um $LUM \
      --fields $F --threads $TH ${SNAPX:-} || return 1
    [ "$DRY" = 1 ] || log "snap: $(grep '^wrap-snap: solve' $O/logs/snap.log | tail -1)"
  else log "snap: off (SNAP=0)"; fi
  [ "$DRY" = 1 ] || { [ "${CLEAN:-1}" = 1 ] && { find $F -mindepth 1 -maxdepth 1 ! -name fields.json -exec rm -rf {} +; rm -rf $O/slabs/s???/fit; }; }
  return 0
}
strip() { local a=$STRIPARGS; [ -e $O/state/strip.args ] && a=$(cat $O/state/strip.args)
          [ "$DRY" = 1 ] || rm -rf $O/strip/strip; X $O/logs/strip.log $U strip $(wrapsdir) $O/strip/strip --threads $TH $a; }
stripflat() { X $O/logs/stripflat.log $U strip-flatten $O/strip/strip --decim ${DECIM:-4} --threads $TH --flip-rows 1 --cg 60 --tol 3e-4 ${FLATX---gpu 1}; }
# stripimg: the filled stripe renders + two mosaics (the pods' strip.sh recipe): a contrast stretch from the 3 middle stripes (rendered
# with fill into strip/render_fill, reused), then scripts/strip/strip_refill.sh: every stripe rendered with FILLX, strip-mosaic img_fill
# (rgb CT + ink) and img_pap (papyrus look, --papyrus-dir, same reads), renders removed after the mosaic. UFSM_RENDER_GPU_MEM 4 (GB):
# the render's device chunk pool stays small enough for a slab fit sharing the GPU (pods, two scrolls); RGPU: CUDA device of the renders
FILLX=${FILLX:---fill 4 --fill-grid 50000 --fill-inpaint 4 --fill-pm 1}
stripimg() {
  local ST=$O/strip RF=$O/strip/render_fill S N k s
  local E="UFSM_ZCACHE_GB=$ZCACHE_GB UFSM_ZCACHE_ADMIT=${ZCACHE_ADMIT:-0.6} UFSM_JPEG_THREADS=$JPEGT UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4} UFSM_RENDER_MEM=${UFSM_RENDER_MEM:-16} ${RGPU:+CUDA_VISIBLE_DEVICES=$RGPU}"
  if [ "$DRY" = 1 ]; then echo "    $E: stretch from the 3 middle stripes (render $FILLX + strip-mosaic --stretch-only), then U=$U CTR=$R CTK=$K/$LV INK=$INK INKK=$LUM PAPMOS=$ST/img_pap bash scripts/strip/strip_refill.sh $ST"; return 0; fi
  mkdir -p $RF $ST/fillmap $ST/logs; S=($(ls $ST/strip/flat/stripes)); N=${#S[@]}; [ $N -gt 0 ] || return 1
  if [ ! -s $ST/stretch.txt ]; then
    for k in $((N / 2 - 1)) $((N / 2)) $((N / 2 + 1)); do [ $k -ge 0 ] && [ $k -lt $N ] || continue; s=${S[$k]}; [ -e $RF/$s.done ] && continue
      mkdir -p $RF/in_$s; ln -sfn $ST/strip/flat/stripes/$s $RF/in_$s/$s
      env $E /usr/bin/time -v nice -n 10 $U render $RF/in_$s $RF --volume $INK $LUM --ct $R $K/$LV --depth 1 --scale 1 --threads ${RENDER_NT:-$TH} $FILLX > $ST/logs/refill_$s.log 2>&1 || return 1
      [ -e $RF/${s}_fill.tif ] && mv $RF/${s}_fill.tif $ST/fillmap/; rm -r $RF/in_$s; touch $RF/$s.done
    done
    nice -n 10 $U strip-mosaic $ST/strip/flat $RF $ST/img_fill --stretch-only 1 --threads 3 2> $ST/logs/stretch.log | awk '/^stretch/ {print $2}' > $ST/stretch.txt
    [ -s $ST/stretch.txt ] || { log "stripimg: no stretch (logs in $ST/logs)"; return 1; }; log "stripimg: stretch $(cat $ST/stretch.txt)"
  fi
  # RJOBS: 2 stripe renders at once when >= 64 GB is available (0490B: 394 -> 355 s, 6.7 -> 18.4 GB RSS, same images), else 1
  local rj=${RJOBS:-$([ $(availgb) -ge 64 ] && echo 2 || echo 1)}
  env $E U=$U CTR=$R CTK=$K/$LV INK=$INK INKK=$LUM RJOBS=$rj RENDER_NT=${RENDER_NT:-$TH} MOS_NT=${MOS_NT:-4} MAXR=${MAXR:-6} MEMGB=${RMEMGB:-0} \
    PAPMOS=$ST/img_pap PGAIN=${PGAIN:-1} PAPPX=${PAPPX:-8000} FILLX="$FILLX" /usr/bin/time -v bash $REFILL $ST >> $O/logs/stripimg.log 2>&1 &&
    [ -e $ST/img_fill/overview.jpg ] && log "stripimg: $(grep '^strip-mosaic: .* overview' $ST/logs/mosaic_fill.log | head -1)"
}
# --- per-wrap renders (p4full3.sh) and faces ---
pw_one() {   # pw_one WRAP GPU (a background subshell: the cache env stays in it)
  zcon; local w=$1 gpu=$2 L=$O/logs/pw/$1 nv dc d=$O/r/$1
  nv=$(stat -c %s $O/wraps/$w.sfc); dc=$(awk -v n=$nv -v m=1500000 'BEGIN { d = 1; while (n / (d * d) > m) d++; print d }')
  if [ ! -e $O/flat/$w/x.tif ]; then
    nice -n 10 $U tifxyz-scale $O/wraps $O/orig 1 --wraps $w > $L.scale.log 2>&1 || return 1
    rm -rf $O/flat/$w
    CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $U flatten $O/orig $O/flat --wraps $w --decim $dc --threads ${FTH:-4} --crop none --gpu 1 > $L.flat.log 2>&1 || return 1
  fi
  nice -n 10 $U tifxyz-scale $O/flat $O/up 1 --wraps $w --flip-rows 1 > /dev/null 2>&1 || return 1
  mkdir -p $d/in; ln -sfn $O/up/$w $d/in/$w
  UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4} CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $U render $d/in $d --volume $INK $LUM --ct $R $K/$LV --depth 1 --scale 1 --threads ${RTH:-8} > $d/render.log 2>&1 || return 1
  nice -n 10 $U ink-compare $O/img/${w}_full_side.jpg --ct $d/${w}_ct.tif --ink "ift3=$d/$w.tif" --px-um $LUM --stretch --stretch-ct --max-px 65000 --threads ${RTH:-8} \
    --then $O/prev/${w}_side.jpg --max-px 8000 > $d/stats.txt 2>&1
  [ -e $O/img/${w}_full_side.jpg ] && rm -f $d/$w.tif $d/${w}_ct.tif
  echo "$(date '+%F %T') $w $(grep -E '^[^ ]+: input' $L.flat.log | head -1 | grep -oE '= [0-9.]+ cm2')" >> $O/logs/perwrap.list
}
perwrap() {
  [ "$DRY" = 1 ] && { echo "    per wrap (largest first, SPAR ${SPAR:-4}, GPUs of FITGPUS in turn): tifxyz-scale -> flatten --gpu 1 -> tifxyz-scale --flip-rows 1 -> render (ink + CT) -> ink-compare --then"; return 0; }
  mkdir -p $O/orig $O/flat $O/up $O/r $O/img $O/prev $O/logs/pw
  local w k=0 rc=0 pg=($FITGPUS)
  for w in $(for f in $O/wraps/w*.sfc; do w=$(basename $f .sfc); [ -e $O/img/${w}_full_side.jpg ] || echo "$(stat -c %s $f) $w"; done | sort -rn | awk '{print $2}'); do
    stopped && break
    while [ $(jobs -r | wc -l) -ge ${SPAR:-4} ]; do sleep 5; done; memok
    pw_one $w ${pg[$((k % ${#pg[@]}))]} & k=$((k + 1))
  done; wait
  echo "perwrap: $(ls $O/img/*_full_side.jpg 2>/dev/null | wc -l) wraps rendered" >> $O/logs/perwrap.log
}
faces() {
  [ "$DRY" = 1 ] && { echo "    ufsm faces per flattened wrap ($O/flat) -> $O/faces (faces_p4e.sh recipe)"; return 0; }
  mkdir -p $O/faces; local w; zcon
  for w in $(ls $O/flat); do [ -e $O/flat/$w/x.tif ] || continue; [ -e $O/faces/out/${w}_recto ] && continue; stopped && break; memok
    X $O/logs/faces.log $U faces $O/flat $O/faces/out --wraps $w --ct $R $K/$LV --ct-um $LUM --threads ${RTH:-8} || echo "faces $w failed" >> $O/logs/faces.log
  done
}
upload() {
  local cmd="ssh desk ~/reg-upload/upcorpus.sh $S   # + lftp mirror -R of wraps / strip img (scripts/corpus/upcorpus.sh)"
  case " $PRIZE " in *" $S "*) log "upload: $S is prize-eligible: NOT uploaded (ask first)"; return 0;; esac
  if [ "${UPLOAD:-0}" != 1 ]; then log "upload: not run (UPLOAD=1 to run): $cmd"; return 0; fi
  X $O/logs/upload.log bash $REPO/scripts/corpus/upcorpus.sh $S
}

# --- the schedule ---
[ "$DRY" = 1 ] && echo "unroll $S: O=$O U=$U ($R $K level $LV, $LUM um; box ${BOX:--}, zr ${ZR:--}, strip args '${STRIPARGS}')"
stage prep prep || exit 1
stage fields fields || exit 1
# the ink (GPU $INKGPU) beside the CPU ref and the GPU-0 slab lane; the per-wrap renders beside the strip
( stage ink ink || touch $O/state/ink.failed ) &
INKP=$!
[ "$DRY" = 1 ] && wait $INKP
{ [ "${CTCAL:-0}" != 1 ] || stage ctcal ctcal; } && stage ref ref && stage slabs slabs && stage join join && stage check check && stage wraps wraps && stage trim trim && stage snap snap ||
  { log "unroll: stopped before the strip"; wait; exit 1; }
if [ "${PERWRAP:-0}" = 1 ]; then ( waitfor ink && stage perwrap perwrap && { [ "${FACES:-0}" = 1 ] && stage faces faces; true; } ) & PWP=$!; [ "$DRY" = 1 ] && wait $PWP; fi
stage strip strip && stage stripflat stripflat && waitfor ink && stage stripimg stripimg || log "unroll: the strip stopped"
wait
stage upload upload
[ "$DRY" = 1 ] || log "unroll $S finished: $(for f in $O/state/*.done; do [ -e $f ] && printf '%s %s; ' $(basename $f .done) "$(cut -d, -f1 $f)"; done)"

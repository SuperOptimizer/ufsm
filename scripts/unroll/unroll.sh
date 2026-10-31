#!/usr/bin/env bash
# unroll.sh SCROLL: one whole-scroll unroll (the Paris 4 coarse-to-fine recipe) as ONE resumable run on one host (the desk: 2 GPUs, 32
# threads, 182 GB). Output root O=$UR/SCROLL (UR default /vesuvius/ufsm/unroll), names as in the Paris 4 run:
#   box.txt axis.json | fields/wind96 | ref-cf | slabs/sNNN + slabs.txt | joined | wraps/wNNN.sfc | ink/$INKNAME.zarr |
#   strip/{strip,render,img} (scripts/strip/strip_run.sh layout) | per-wrap (PERWRAP=1, p4full layout): orig flat up r img prev |
#   faces/ (FACES=1) | state/<stage>.done (wall s, CPU s, max RSS) | logs/<stage>.log | progress.log
# Stages and their order (critical path first; -> = waits for):
#   prep    box (scrolls.tsv or the scanprep mask) + axis (scrolls.tsv file or ufsm axis)
#   fields  wind96 fit fields: the box split in z at a 1024-voxel boundary, one predict per GPU (same global tile grid), joined by
#           fields_cat.sh (shard files moved: the result is the single-GPU run's); FIELDGPUS="0" = one predict
#   ink     -> prep only; GPU INKGPU (1) from when the fields are done: INKCMD (the ink agent's recipe; default: the corpus 2-pass
#           ift3 ink, frozen GN, window 640 / 64 + a +256 offset pass, ink-blend) -> ink/$INKNAME.zarr
#   ref     -> fields: count-field --model-only (REFCMD overrides; CPU, TH threads)
#   slabs   -> ref: snapped slab fits, one lane per GPU in FITGPUS ("0 1"); the INKGPU lane starts once the ink is done
#   join    -> slabs: fit-join --ref ;  wraps -> join: ufsm wraps (.sfc, step 4, --zr from scrolls.tsv, --min-cm2, --col-r p90)
#   strip   -> wraps: ufsm strip ($STRIPARGS) ; stripflat -> strip: strip-flatten (FLATX, default "--gpu 1"; FLATX= for the CPU) ; stripimg -> stripflat + ink:
#           stripe renders + strip-mosaic (strip_run.sh)
#   perwrap -> wraps + ink (PERWRAP=1): flatten (GPU) + render + ink-compare per wrap, largest first (p4full3.sh recipe)
#   faces   -> perwrap (FACES=1): ufsm faces per flattened wrap
#   upload  never automatic: UPLOAD=1 and the scroll not prize-eligible (feedback: ask first); prints the command otherwise
# Heavy CPU stages wait for MemAvailable >= MINMEM (40) GB. STOP: touch $O/STOP (checked between stages / slabs / wraps). A stage
# whose done marker exists is skipped; partial stages resume (predict --resume parts, slab claims, per-wrap outputs).
# DRY=1: print the plan (commands) without running anything. Runs from a copy of these scripts in $O/bin (edits never touch a run).
# Env: U (ufsm), UR, TH (20), FIELDGPUS ("0 1"), FITGPUS ("0 1"), INKGPU (1), INKCMD, REFCMD, CK, INKCK, INKNAME, MINMEM, PERWRAP,
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
  mkdir -p $O/bin && cp $H/unroll.sh $H/fields_cat.sh $H/scrolls.tsv $REPO/scripts/wbench/mkcrop.sh $REPO/scripts/strip/strip_run.sh $O/bin/ || exit 1
  UNROLL_COPY=1 UNROLL_REPO=$REPO exec bash $O/bin/unroll.sh "$@"
fi
MKCROP=$H/mkcrop.sh; [ -e $MKCROP ] || MKCROP=$REPO/scripts/wbench/mkcrop.sh
STRIPRUN=$H/strip_run.sh; [ -e $STRIPRUN ] || STRIPRUN=$REPO/scripts/strip/strip_run.sh

[ "$DRY" = 1 ] || mkdir -p $O/state $O/logs
# the shared decoded-CT cache (src/zshm.c) for the render / faces / ink-compare stages: ZCACHE_GB (auto: 1/4 of the available RAM, at
# most 30 GB; off on the laptop = WSL), admit 0.6 (P4 w137 / w138 re-renders: 47-54% hits, a render 32.8 -> 20.0 s). The cache file
# is removed at the end of the run (also on STOP / failure). strip-mosaic: UFSM_JPEG_THREADS = cores / 2 (it encodes up to 2 stripes at once, --stripe-jobs 2).
if [ "${ZCACHE_GB:-auto}" = auto ]; then
  if grep -qi microsoft /proc/version 2> /dev/null; then ZCACHE_GB=0; else ZCACHE_GB=$(( $(availgb) / 4 )); [ $ZCACHE_GB -gt 30 ] && ZCACHE_GB=30; fi
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

ref() {
  [ "$DRY" = 1 ] || rm -rf $REF
  if [ -n "${REFCMD:-}" ]; then X $O/logs/ref.log $REFCMD $F $REF; else
  X $O/logs/ref.log $U count-field $F $O/cf.unused --R 0 --pool 8 --phase-w 4 --iters 1200 --model $REF --model-only 1 --model-stride ${MSTRIDE:-4} \
    --threads $TH --mem-gb ${CFMEM:-24}; fi && { [ "$DRY" = 1 ] || [ -e $REF/model.json ]; }
}

# --- ink (another agent's recipe via INKCMD SCROLLDIR GPU OUT; default the corpus 2-pass) ---
ink() {
  [ "$DRY" = 1 ] || mkdir -p $O/ink
  if [ -n "${INKCMD:-}" ]; then X $O/logs/ink.log $INKCMD $O $INKGPU $INK; return; fi
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
    env UFSM_FIT_THREADS=10 /usr/bin/time -v nice -n 5 $U fit $d/f --gpu $g --out $d/fit --dir $dir --ref-model $REF --ref-snap 1 --uw-repair 0 ${FITX:-} > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s" >> $O/slabs.txt
    if [ $rc = 0 ]; then touch $d/done; elif grep -q "no data for the prior" $d/fit.log; then touch $d/empty $d/done; fi
    rm -rf $d/f; exec 9>&-
  done
}
slabs() {
  local g pids=()
  for g in $FITGPUS; do
    if [ "$g" = "$INKGPU" ] && ! done_ ink; then { [ "$DRY" = 1 ] && echo "    (GPU $g lane starts once the ink is done)" || while ! done_ ink && ! stopped; do sleep 30; done; lane $g; } &
    else lane $g & fi; pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  [ "$DRY" = 1 ] && return 0
  local ns=$(( ($(jq -r '.shape[0]' $F/fields.json) + I - 1) / I )) nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l)
  cat $O/slabs/s???/fit.log | grep -E 'User time|System time|Maximum resident' >> $O/logs/slabs.log   # (the stage marker's CPU s)
  log "slabs $nd of $ns done"; [ $nd = $ns ]
}
join() {
  local args=() d c0 c1 a b
  if [ "$DRY" = 1 ]; then X $O/logs/join.log $U fit-join $O/joined $F --blend 16 --ref $REF "slabs/sNNN/fit 2c0 2c1 ..."; return; fi
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; [ -e $d/empty ] && continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; X $O/logs/join.log $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}"
}
wraps() {
  local zr=; [ -n "$ZR" ] && [ "$ZR" != - ] && zr="--zr $ZR"
  [ "$DRY" = 1 ] || rm -rf $O/wraps
  X $O/logs/wraps.log $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH $zr --min-cm2 ${MINCM2W:-0.5} || return 1
  [ "$DRY" = 1 ] && return 0
  [ "${CLEAN:-1}" = 1 ] && { find $F -mindepth 1 -maxdepth 1 ! -name fields.json -exec rm -rf {} +; rm -rf $O/slabs/s???/fit; }   # (the joined model is standalone)
  return 0
}
strip() { [ "$DRY" = 1 ] || rm -rf $O/strip/strip; X $O/logs/strip.log $U strip $O/wraps $O/strip/strip --threads $TH $STRIPARGS; }
stripflat() { X $O/logs/stripflat.log $U strip-flatten $O/strip/strip --decim ${DECIM:-4} --threads $TH --flip-rows 1 --cg 60 --tol 3e-4 ${FLATX---gpu 1}; }
stripimg() {   # stripe renders + mosaic: strip_run.sh skips its strip / flatten stages (done here)
  [ "$DRY" = 1 ] && { echo "    UFSM_ZCACHE_GB=$ZCACHE_GB UFSM_JPEG_THREADS=$JPEGT U=$U CTR=$R CTK=$K/$LV INK=$INK INKK=$LUM bash $STRIPRUN $O/strip $O/wraps"; return 0; }
  UFSM_ZCACHE_GB=$ZCACHE_GB UFSM_ZCACHE_ADMIT=${ZCACHE_ADMIT:-0.6} UFSM_JPEG_THREADS=$JPEGT \
    U=$U CTR=$R CTK=$K/$LV INK=$INK INKK=$LUM NT=$TH RENDER_NT=${RENDER_NT:-8} /usr/bin/time -v bash $STRIPRUN $O/strip $O/wraps >> $O/logs/stripimg.log 2>&1
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
  UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 nice -n 10 $U render $d/in $d --volume $INK $LUM --ct $R $K/$LV --depth 1 --scale 1 --threads ${RTH:-8} > $d/render.log 2>&1 || return 1
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
stage ref ref && stage slabs slabs && stage join join && stage wraps wraps || { log "unroll: stopped before the wraps"; wait; exit 1; }
if [ "${PERWRAP:-0}" = 1 ]; then ( waitfor ink && stage perwrap perwrap && { [ "${FACES:-0}" = 1 ] && stage faces faces; true; } ) & PWP=$!; [ "$DRY" = 1 ] && wait $PWP; fi
stage strip strip && stage stripflat stripflat && waitfor ink && stage stripimg stripimg || log "unroll: the strip stopped"
wait
stage upload upload
[ "$DRY" = 1 ] || log "unroll $S finished: $(for f in $O/state/*.done; do [ -e $f ] && printf '%s %s; ' $(basename $f .done) "$(cut -d, -f1 $f)"; done)"

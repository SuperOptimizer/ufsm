#!/usr/bin/env bash
# unroll.sh SCROLL: one whole-scroll unroll (the Paris 4 coarse-to-fine recipe) as ONE resumable run on one host (the desk: 2 GPUs, 32
# threads, 182 GB). Output root O=$UR/SCROLL (UR default /vesuvius/ufsm/unroll), names as in the Paris 4 run:
#   box.txt axis.json | fields/wind96 | ref-cf | slabs/sNNN + slabs.txt | joined | wraps/wNNN.sfc | ink/$INKNAME.zarr |
#   strip/{strip,img_fill,img_pap,fillmap,stretch.txt} (scripts/strip/strip_refill.sh layout) | per-wrap (PERWRAP=1, p4full layout): orig flat up r img prev |
#   faces/ (FACES=1) | state/<stage>.done (wall s, CPU s, max RSS) | logs/<stage>.log | progress.log
# Stages and their order (critical path first; -> = waits for):
#   prep    box (scrolls.tsv or the scanprep mask) + axis (scrolls.tsv file or ufsm axis)
#   fields  wind96 fit fields (window 384 x 640^2 / 64; FWIN=auto: predict --window auto, opt-in): the box split in z at a 512-voxel boundary, one predict per GPU (same global tile grid), joined by
#           fields_cat.sh (shard files moved: the result is the single-GPU run's); FIELDGPUS="0" = one predict. A latent-grid CK (FKIND auto: the
#           checkpoint header has "lat_c"; FKIND=lat / vox forces it): fields/lat from cube windows LWIN / LHALO (640 / 64, the gate's U1
#           settings: stride 512, so the 512-voxel z parts keep one tile grid), the latent heads' fit planes (winding mod 4 too); --fit-h 2 runs
#           the slab plane path (the fused nn_fit_accum_lat kernel takes --fit-h 8 only)
#   ink     -> prep only; GPU INKGPU (1): INKAT=slabs (default with 2+ fit GPUs; one-GPU runs: INKAT=fields INKWAIT=0) its GroupNorm calibration (inkcal) beside the ref, the main pass from
#           the end of the slabs (beside join .. stripflat, all CPU); INKAT=fields: all of it from when the fields are done: INKCMD (the ink agent's recipe; default: the feathered
#           single pass of inklib.sh: ift3, frozen GN, --window auto --halo 32 --feather 32, resumable) -> ink/$INKNAME.zarr
#   ctcal   -> fields (CTCAL=1 only, OFF by default): ufsm ct-cal --ct-um $LUM (the CT LEVEL's um) -> ct.cal; ref adds --ct-cal
#   rectosp -> fields (default ON when the scroll has m7 surface predictions on the mirror; RECTOSP=0 off): ufsm recto-sp $F
#           fields/rsp --src <m7 prob> --half-um 400 --half2-um 800 --blend 0.3 (RECTOSPX overrides the flags; RECTOSRC="ROOT KEY UM"
#           names the source): the sheet spacing blended with the m7 recto hit spacing (RECTO 11-02: whole P4 ids .1764 -> .2054, 1667 slabs
#           +.006..+.045, 0139 -.01..-.02); the ref, slabs and later stages then read fields/rsp. No m7 for this scan + level: skipped.
#   ref     -> fields: count-field --model-only (REFCMD F REF [--ct-cal FILE] overrides; cf GPU, TH threads) REFX: extra count-field flags (e.g. --rho-scale S)
#   slabs   -> ref: snapped slab fits, FITLANES lanes per GPU in FITGPUS ("0 1"; auto: by device memory, 1..3); the INKGPU lanes start once the ink is done; then the
#           bad-slab check (fit.log: "illegal memory", "loss 0.00000", no final loss line): refit on 1 lane, up to 3 tries, else fail
#   join    -> slabs: fit-join --ref (refuses bad slabs) ; check -> join: wrap-check (wrapcheck.tsv verdicts; 0 CT peaks = fail) +
#           unroll-qa (label-free identity gates, warn only; UQSTRICT=1 fatal; FIBRES=UFIB adds the fibre gate; qa/unroll-qa)
#           check also runs cf-demod on the ref (DRIFT=1: the label-free z drift of the sheet identity, ALARM > DRIFTWARN 0.75 turn,
#           DRIFTMAX fails) and, with GTBOXES + GTCODES, scores joined / ref-cf on GT boxes (scripts/spiralqa/ugeom/gtboxes.sh, CPU).
#           DEMOD=1 (off): no slab fits; join = cf-demod --out of the ref (the ref's integers, the phase's sub-turn; DEMODR nodes)
#   wraps   -> check: ufsm wraps (.sfc, step 4, --zr from scrolls.tsv, --min-cm2, --col-r p90)
#   trim    -> wraps: from both ends, wraps beyond the wrap-check table, with collapsed gaps (rnet_med < RNETMIN 0.6) or near-empty (on-papyrus
#           <= PAPMIN 0.1; surf-stats) dropped: strip --first / --last = the first kept ids (state/strip.args; per-wrap inputs state/trim.tsv,
#           the dropped ids in progress.log); TRIM=0 off; a --last / --first in STRIPARGS keeps that end (a tsv --last is ignored)
#   snap    -> trim (SNAP=1 default; SNAP=0 off): ufsm wrap-snap wraps -> wraps_snap: each wrap moved along its rays onto the sheets
#           (smooth per-wrap offsets, order kept: never past half the gap to its neighbours, never into masked CT; SNAPT target, SNAPX
#           extra flags); the strip and its renders then use wraps_snap. CLEAN runs after it (the phase / mix targets read the fields)
#           MASKMAX (off): fail when more than that share of the wrap vertices lie on masked CT (wrap-snap --masked-max)
#   strip   -> snap: ufsm strip ($STRIPARGS + --last) ; stripflat -> strip: strip-flatten (FLATX, default "--gpu 1"; FLATX= for the CPU) ;
#   distort -> stripflat: strip-distort (the flattening's stretch / folds) as a gate: dist p95 <= DISTP95 (0.35), folded <= DISTFOLD (0.005), no
#           skipped strip-flatten chunk, else the stage fails (DISTGATE=0: logged only); overlays in strip/distort
#   stripimg -> stripflat + ink: stretch from the 3 middle stripes, filled stripe renders (FILLX) + strip-mosaic img_fill (rgb) and
#           img_pap (papyrus look) via scripts/strip/strip_refill.sh; renders UFSM_RENDER_GPU_MEM=4 (a fit may share the GPU), RGPU
#   perwrap -> wraps + ink (PERWRAP=1): flatten (GPU) + render + ink-compare per wrap, largest first (p4full3.sh recipe)
#   faces   -> perwrap (FACES=1): ufsm faces per flattened wrap
#   upload  never automatic: UPLOAD=1 and the scroll not prize-eligible (feedback: ask first); prints the command otherwise
# UPTO=STAGE: stop after that stage of the critical path (e.g. UPTO=slabs for a GPU window: fields, ink, slabs; then re-run without it, the
# GPUs hidden: CUDA_VISIBLE_DEVICES= UFSM_RENDER_GPU=0 FLATX=); NOINK=1: no ink stage (a comparison run up to check; the GPU lane of
# INKGPU does not wait for it)
# Heavy CPU stages wait for MemAvailable >= MINMEM (40) GB. STOP: touch $O/STOP (checked between stages / slabs / wraps). A stage
# whose done marker exists is skipped; partial stages resume (predict --resume parts, slab claims, per-wrap outputs).
# DRY=1: print the plan (commands) without running anything. Runs from a copy of these scripts in $O/bin (edits never touch a run).
# Env: RECTOSP (1; 0 = no m7 recto spacing), RECTOSRC, RECTOSPX, U (ufsm), UR, TH (20), FIELDGPUS ("0 1"), FITGPUS ("0 1"), FITLANES (auto), INKGPU (1), INKCMD, REFCMD, CK, INKCK, INKNAME, MINMEM, PERWRAP,
#   INKPX (extra ink predict args, e.g. "--channel ink" for a v2 universal checkpoint as INKCK),
#   CTROOT / CTKEY (replace the table's CT root / key), ZCACHE_GB (auto) / ZCACHE_ADMIT (0.6): the decoded-CT cache of the render stages,
#   TAG (output dir suffix), ITH / PTH (ink / fields predict threads), FACES, SPAR (4 per-wrap jobs), CLEAN (1: fields + slab fits deleted once the wraps exist), STRIPARGS / ZR / BOX (override scrolls.tsv).
set -u
S=${1:?usage: unroll.sh SCROLL}
H=$(cd "$(dirname "$0")" && pwd)
UR=${UR:-/vesuvius/ufsm/unroll}; O=$UR/$S${TAG:-}   # (TAG: e.g. a test crop beside the full run)
U=${U:-/vesuvius/ufsm/unroll/bin/ufsm}
CK=${CK:-/vesuvius/ufsm/podready/pod/ckpt/wind96-wk1-best.ckpt}; INKCK=${INKCK:-/vesuvius/ufsm/ink-ft/runs/ift3/snap/ift3-24000.ckpt}   # (CK: the gate's wind96 copy, sha1 dafb8233; /vesuvius/ufsm/paris4 is gone since the 11-12 cleanup)
# host limits (cgroup-aware on pods): ufsm sysinfo --ncpu / --avail-gb when the binary has it, else nproc / /proc/meminfo
# (no sysinfo on a pod, i.e. a cgroup with a memory.max: the cgroup's own cpu.max / memory.max, never nproc / MemAvailable, which show the
#  host there: 256 CPUs / 755 GB on a ~62 CPU / 200 GB pod; mini audit podscripts #7)
cgmax() { local f=/sys/fs/cgroup/$1; [ -r $f ] && [ "$(cut -d' ' -f1 $f)" != max ] && cat $f; }
ncpu() { local n; n=$($U sysinfo --ncpu 2> /dev/null) && [ "$n" -gt 0 ] 2> /dev/null && { echo $n; return; }
         n=$(cgmax cpu.max) && { echo $n | awk '{print int($1 / $2) > 0 ? int($1 / $2) : 1}'; return; }; nproc; }
availgb() { local g; g=$($U sysinfo --avail-gb 2> /dev/null) && [ "$g" -ge 0 ] 2> /dev/null && { echo $g; return; }
            g=$(cgmax memory.max) && { awk -v m=$g '/^anon |^shmem / {u += $2} END {print int((m - u) / 1073741824)}' /sys/fs/cgroup/memory.stat; return; }
            awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
# the ink: INKMODE lat (default since 2026-11-14, the latent-ink-only policy): scripts/pod/latink.sh (two tile grids + ink-blend, ink / c12 /
# c13 heads, --axis) with LATINK_CK (default the newest inkpod snap, 15000) as INKCMD; INKMODE ift3: the legacy feathered single pass of INKCK
INKMODE=${INKMODE:-lat}
if [ "$INKMODE" = lat ] && [ -z "${INKCMD+x}" ]; then
  export LATINK_CK=${LATINK_CK:-/vesuvius/ufsm/v2/lat/runs/inkpod/snap/15000.ckpt} U; INKCMD="bash $([ -e $H/latink.sh ] && echo $H || echo $H/../pod)/latink.sh";   # (the run's copy in $O/bin, the repo's on DRY)
  INKNAME=${INKNAME:-inkpod-$(basename $LATINK_CK .ckpt)}
fi
INKNAME=${INKNAME:-ift3-24000}; TH=${TH:-$(ncpu)}; [ "$TH" -gt 20 ] 2> /dev/null && [ -z "${TH_ALL:-}" ] && TH=20; MINMEM=${MINMEM:-40}; INKGPU=${INKGPU:-1}
FIELDGPUS=${FIELDGPUS:-0 1}; FITGPUS=${FITGPUS:-0 1}; DRY=${DRY:-0}; I=${SLABI:-64}; OV=${SLABOV:-16}
# SGPU: the device of every stage that takes no --gpu of its own (ref / refstream count-field, strip-flatten, the strip renders, wraps ..),
# the first of FITGPUS by default: they ran on CUDA device 0 whatever FITGPUS said (2026-11-17: with FITGPUS=1 on the desk, the ref's
# pooling, strip-flatten's SLIM and the stripimg renders went to card 0, the training card; its 0.7 GB free sent the ref's CG and the
# renders to the CPU and strip-flatten took 204 s). An explicit CUDA_VISIBLE_DEVICES (also empty: GPUs hidden) is kept as given.
SGPU=${SGPU:-${FITGPUS%% *}}
# GVIS: without a CUDA_VISIBLE_DEVICES of the caller's, every ufsm call of the run sees only the cards it was given (FIELDGPUS, FITGPUS,
# INKGPU, SGPU, RGPU; lead 2026-11-17: a predict on --gpu 1 still opened a 128 MiB context on card 0, the training card): the run
# exports CUDA_VISIBLE_DEVICES=GVIS, --gpu takes the card's index in it (lg), per-job CUDA_VISIBLE_DEVICES=<card> stay physical ids
# (UNROLL_GVIS: the run's copy in $O/bin is exec'd with the CUDA_VISIBLE_DEVICES set here; it keeps the same GVIS)
GVIS=
if [ -n "${UNROLL_GVIS:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "${UNROLL_GVIS// /,}" ]; then GVIS=$UNROLL_GVIS
elif [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
  for g in $FIELDGPUS $FITGPUS $INKGPU $SGPU ${RGPU:-}; do case " $GVIS " in *" $g "*) ;; *) GVIS="$GVIS $g";; esac; done
  GVIS=$(echo $GVIS); export CUDA_VISIBLE_DEVICES=${GVIS// /,} UNROLL_GVIS=$GVIS
fi
lg() { local i=0 g; [ -n "$GVIS" ] && for g in $GVIS; do [ "$g" = "$1" ] && { echo $i; return; }; i=$((i + 1)); done; echo $1; }
sgpu() { [ -n "$GVIS" ] && [ -n "$SGPU" ] && echo "CUDA_VISIBLE_DEVICES=$SGPU"; return 0; }
# SLABI / SLABOV: slab cells / overlap (default 64 / 16). A full-section Paris 3 slab (96 x 3680 x 4768 fields, 16.8 GB) does not fit a 16 GB GPU and
# runs from managed memory at 20-60 min per slab (desk 5060 Ti, 11-01); SLABI=32 (64 rows, 11.2 GB) fits the device.
PRIZE="PHerc0125 PHerc0175A PHerc0175B PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0306B PHerc0358 PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0800 PHerc0813 PHerc0826 PHerc0846A PHerc0846B PHerc1203 PHerc1218 PHerc1545"

# the scan: scrolls.tsv (box / zr / strip args), else the corpus list.tsv (box from the mask)
row=$(awk -F'\t' -v s=$S '$1 == s' $H/scrolls.tsv)
REPO=${UNROLL_REPO:-$(cd $H/../.. && pwd)}
[ -n "$row" ] || row=$(awk -F'\t' -v s=$S '$1 == s { print $0 "\t-\t-\t-" }' $REPO/scripts/corpus/list.tsv)
[ -n "$row" ] || { echo "unroll: $S is in neither scrolls.tsv nor corpus/list.tsv"; exit 2; }
IFS=$'\t' read -r _ R K UM LV AXS TBOX TZR TSTRIP <<< "$row"
R=${CTROOT:-$R}; K=${CTKEY:-$K}; AXS=${AXIS:-$AXS}   # (pods: the CT mirrored to local disk; AXIS: an axis file other than the table's)
LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }')
# CTCACHE (default $O/ctcache; 0 off): an HTTP CT is cached chunk by chunk on local disk for every stage (UFSM_HTTP_CACHE, src/zarr3.c):
# the second fields grid, the checks, the trim, the snap and the renders read it locally, and the predicts prefetch into it (0332 32-window
# box: 34.4 s without, 23.0 s cold, 19.7 s warm; outputs byte-identical). The default directory is deleted when the run finishes (KEEPCT=1 keeps it)
case $R in http://*|https://*) [ "${CTCACHE:-}" != 0 ] && export UFSM_HTTP_CACHE=${CTCACHE:-$O/ctcache};; esac
BOX=${BOX:-$TBOX}; ZR=${ZR:-$TZR}; [ "$TSTRIP" = - ] && TSTRIP=
# a wrap id (--last N) is in ONE run's gauge only (P3 11-02: the prod run's --last 432 skipped the trim of a re-unroll numbered w025-w266):
# a --last in scrolls.tsv / list.tsv is dropped here (the trim stage finds it per run); only a STRIPARGS env override keeps one (checked in trim)
if [ -z "${STRIPARGS:-}" ]; then STRIPARGS=$(echo " $TSTRIP " | sed -E 's/ --last [0-9]+ / /; s/^ +//; s/ +$//'); else [ "$STRIPARGS" = - ] && STRIPARGS=; fi
FKIND=${FKIND:-auto}; [ "$FKIND" = auto ] && { head -c 8192 "$CK" 2> /dev/null | grep -q '"lat_c"' && FKIND=lat || FKIND=vox; }
FNAME=$([ "$FKIND" = lat ] && echo lat || echo wind96)
F=$O/fields/$FNAME; REF=$O/ref-cf; INK=$O/ink/$INKNAME.zarr

# the run's own copy of the scripts (bash reads scripts incrementally: editing the repo must not change a running driver)
if [ "$DRY" != 1 ] && [ "${UNROLL_COPY:-0}" != 1 ]; then
  mkdir -p $O/bin && cp $H/unroll.sh $H/fields_cat.sh $H/scrolls.tsv $H/spiral_dir.tsv $REPO/scripts/wbench/mkcrop.sh $REPO/scripts/strip/strip_run.sh $REPO/scripts/strip/strip_refill.sh $REPO/scripts/pod/latink.sh $O/bin/ || exit 1
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
# memok: availgb is the cgroup headroom under a memcap scope (sysinfo --avail-gb), so MINMEM is checked against the cap, not the host:
# capped at a quarter of the smallest memory.max on this process's cgroup path (11-12: MINMEM 40 under memcap 40G deadlocked the slab
# lanes in memok for 8 min until the scope was raised to 60G)
capgb() { local cg d m best=; cg=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2> /dev/null); d=/sys/fs/cgroup$cg
          while [ -n "$cg" ] && [ "$d" != /sys/fs/cgroup ] && [ "$d" != / ]; do m=$(cat $d/memory.max 2> /dev/null); [ -n "$m" ] && [ "$m" != max ] && { m=$((m >> 30)); [ -z "$best" ] || [ $m -lt $best ] && best=$m; }; d=$(dirname $d); done
          echo ${best:-}; }
memok() { [ "$DRY" = 1 ] && return 0; local mm=$MINMEM c; c=$(capgb); [ -n "$c" ] && [ $mm -gt $((c / 4)) ] && mm=$((c / 4)); while [ $(availgb) -lt $mm ]; do sleep 30; done; }
# X LOG CMD..: a command, nice 10, timed (/usr/bin/time -v appended to LOG); DRY: printed
X() { local l=$1; shift; if [ "$DRY" = 1 ]; then echo "    $*  > $l"; return 0; fi
      /usr/bin/time -v nice -n 10 "$@" >> $l 2>&1; }
# stage NAME FN: FN unless done; the marker records wall s, then the CPU s and max RSS summed / maxed over the stage's timed commands
stage() {
  local st=$1 fn=$2 t0 rc l=$O/logs/$1.log
  done_ $st && return 0; stopped && return 1
  if [ "$DRY" = 1 ]; then
    case $st in fields|slabs|ink|inkcal|perwrap|faces) echo "[$st] (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}; per-job --gpu / devices)"; $fn;;
      *) if [ -n "$(sgpu)" ]; then echo "[$st] (CUDA_VISIBLE_DEVICES=$SGPU)"; CUDA_VISIBLE_DEVICES=$SGPU $fn; else echo "[$st] (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset})"; $fn; fi;; esac
    return 0; fi
  log "$st start"; t0=$(date +%s); : >> $l; local m0=$(wc -l < $l)
  # (the stages that place themselves: fields / slabs / ink / perwrap pass --gpu or set CUDA_VISIBLE_DEVICES per job; the rest on SGPU)
  case $st in fields|slabs|ink|inkcal|perwrap|faces) $fn;; *) if [ -n "$(sgpu)" ]; then CUDA_VISIBLE_DEVICES=$SGPU $fn; else $fn; fi;; esac; rc=$?
  local cpu rss; read -r cpu rss < <(tail -n +$((m0 + 1)) $l | awk -F': ' '/User time|System time/ {c += $2} /Maximum resident/ {if ($2 > r) r = $2} END {printf "%.0f %.1f\n", c, r / 1048576}')
  if [ $rc = 0 ]; then echo "$(( $(date +%s) - t0 )) s wall, ${cpu:-0} CPU s, ${rss:-0} GB max RSS" > $O/state/$st.done; log "$st done: $(cat $O/state/$st.done)"; mstage $st
  else log "$st FAILED rc $rc (logs/$st.log)"; fi
  return $rc
}
# state/manifest.json: what the run was made from and what it decided (audit 2026-11-14: 14 of 17 u1 runs kept only their images, no
# checkpoint / axis hash, no direction or trim inputs, so the direction default could not be audited afterwards). String values; mset KEY
# VALUE under a lock (the ink stage runs beside the critical path); mstage STAGE after each stage records that stage's decisions.
mset() { [ "$DRY" = 1 ] && return 0; local f=$O/state/manifest.json
  ( flock 8; [ -s $f ] || echo '{}' > $f; jq --arg k "$1" --arg v "$2" '.[$k] = $v' $f > $f.tmp && mv $f.tmp $f ) 8> $O/state/.manifest.lock; }
fsha() { [ -e "$1" ] && sha256sum "$(readlink -f "$1")" | cut -c1-16 || echo missing; }   # (16 hex digits of the sha256; a 200 MB ckpt ~0.5 s)
lastline() { grep -hE "$1" "${@:2}" 2> /dev/null | tail -1; }
mstart() {
  [ "$DRY" = 1 ] && return 0
  mset run_start "$(date '+%F %T')"; mset host "$(hostname)"; mset ufsm "$(readlink -f $U)"; mset ufsm_sha "$(fsha $U)"
  mset repo_commit "$(git -C $REPO rev-parse --short HEAD 2> /dev/null || cat $REPO/COMMIT 2> /dev/null || echo unknown)"
  mset ct "$R $K level $LV ($LUM um)"; mset ck "$(readlink -f $CK)"; mset ck_sha "$(fsha $CK)"; mset fields_kind $FKIND
  if [ -n "${INKCMD:-}" ]; then mset inkcmd "$INKCMD"; mset ink_ck "${LATINK_CK:-the default of INKCMD}"; [ -n "${LATINK_CK:-}" ] && mset ink_ck_sha "$(fsha $LATINK_CK)"
  else mset ink_ck "$(readlink -f $INKCK)"; mset ink_ck_sha "$(fsha $INKCK)"; fi
  mset box "${BOX:-mask}"; mset zr "${ZR:--}"; mset stripargs_env "${STRIPARGS:-}"
  mset env "$(for v in CK INKCK INKCMD LATINK_CK FILLX FLATX SLABFIT SLABI SLABOV REFX REFCMD RECTOSP TRIM PAPMIN RNETMIN DISTGATE DISTP95 DISTFOLD MINCM2W SNAP SNAPT FWIN LWIN LHALO FSKIP FBLEND ONEPASS OPWIN FCTR FCTK FILLCT LATWROUND INKGATE INKMODE; do [ -n "${!v+x}" ] && printf '%s=%s; ' $v "${!v}"; done)"
}
mstage() {
  [ "$DRY" = 1 ] && return 0
  case $1 in
    prep)   mset axis_source "$AXS"; mset axis_sha "$(fsha $O/axis.json)"; mset box "$(cat $O/box.txt 2> /dev/null)";;
    ref)    mset dir "$(jq -r .dir $REF/model.json 2> /dev/null)"; mset ref_line "$(lastline '^count-field: .*dir [+-]' $O/logs/ref.log)"
            mset dir_vote_line "$(lastline 'dir.*(vote|rings|default)' $O/logs/ref.log)";;
    slabs)  mset slab_ref_coherence "$(for d in $O/slabs/s???; do c=$(lastline 'U_ref - phase' $d/fit.log | grep -oE 'coherence [0-9.]+' | cut -d' ' -f2); printf '%s:%s ' ${d##*/} ${c:-na}; done)";;
    join)   mset join_agreement "$(lastline 'integer agreement on the overlaps' $O/logs/join.log | sed 's/^ *//')"; mset join_gauges "$(lastline '^fit-join: gauges' $O/logs/join.log)"
            grep -q 'fit-join: WARNING slab' $O/logs/join.log && log "join: $(grep -c 'fit-join: WARNING slab' $O/logs/join.log) slabs with an unconstrained gauge (no ref, no overlap cells; logs/join.log)";;
    check)  mset wrapcheck "$(lastline '^wrap-check: verdicts' $O/logs/check.log)";;
    trim)   mset strip_args "$(cat $O/state/strip.args)"; mset trim "$(grep -h '^[0-9-]* [0-9:]* trim: ' $O/progress.log | tail -3 | cut -d' ' -f3- | cut -c1-600 | tr '\n' '|')";;
    stripflat) mset stripflat "$(lastline '^strip-flatten: flattened' $O/logs/stripflat.log)";;
    distort) mset distort "$(lastline '^strip-distort: total' $O/logs/distort.log)";;
    stripimg) mset strip "$(lastline '^strip-mosaic: .* length' $O/strip/logs/mosaic_fill.log)";;
  esac
}
# upto STAGE: true when UPTO names STAGE (the schedule stops there: the background ink is waited for)
upto() { [ "${UPTO:-}" = "$1" ] && { [ "$DRY" = 1 ] && echo "(UPTO=$1: stop here)" || log "unroll: UPTO=$1 reached: stopping (re-run without UPTO for the rest)"; wait; exit 0; }; return 0; }
waitfor() { [ "$DRY" = 1 ] && { echo "    (waits for state/$1.done)"; return 0; }; while ! done_ $1; do stopped && return 1; [ -e $O/state/$1.failed ] && return 1; sleep 30; done; }

prep() {
  if [ -n "$BOX" ] && [ "$BOX" != - ]; then [ "$DRY" = 1 ] && echo "    box $BOX" || echo $BOX > $O/box.txt
  else
    local id=$(basename $K | cut -c1-14) m sh GZ GY GX cell
    m=${PREPDIR:-/vesuvius/ufsm/u0/prep}/$S-$id.mask.raw; sh=$(jq -r '.shape | join(" ")' $R/$K/$LV/zarr.json)
    # (a missing mask, or one with no set voxel, fails the stage: od on a missing file fed awk nothing and its END printed a corner box near
    # the origin, exit 0, the whole unroll then ran on it; audit mini/shell #1)
    [ "$DRY" = 1 ] || { [ -s $m ] && [ -s $m.json ]; } || { log "prep: no mask $m (+ .json) and no BOX for $S: refusing"; return 1; }
    read -r GZ GY GX < <(jq -r '.grid | join(" ")' $m.json); cell=$(jq -r .cell $m.json)
    [ "$DRY" = 1 ] && { echo "    box from $m (corpus desk.sh prep)"; } || od -An -v -tu1 -w$((GY * GX)) $m | awk -v X=$GX -v c=$cell -v sh="$sh" '
      { for (i = 1; i <= NF; i++) if ($i > 0) { y = int((i - 1) / X); x = (i - 1) % X; z = NR - 1
          if (!n++) { z0 = z1 = z; y0 = y1 = y; x0 = x1 = x }
          if (z < z0) z0 = z; if (z > z1) z1 = z; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x } }
      END { if (!n) exit 1; split(sh, s, " "); a[1] = z0; a[2] = y0; a[3] = x0; b[1] = z1; b[2] = y1; b[3] = x1
        for (d = 1; d <= 3; d++) { lo = (a[d] - 1) * c; if (lo < 0) lo = 0; lo = int(lo / 512) * 512; hi = (b[d] + 2) * c; if (hi > s[d]) hi = s[d]; o[d] = lo; e[d] = hi - lo }
        printf "%d,%d,%d,%d,%d,%d\n", o[1], o[2], o[3], e[1], e[2], e[3] }' > $O/box.txt || { log "prep: mask $m has no set voxel: refusing"; rm -f $O/box.txt; return 1; }
  fi
  if [ "$AXS" = auto ]; then X $O/logs/prep.log $U axis $R $K $O/axis.json || return 1
  else [ "$DRY" = 1 ] && echo "    axis $AXS" || cp $AXS $O/axis.json; fi
  [ "$DRY" = 1 ] || [ -s $O/box.txt ]
}

# --- fields: one predict per GPU on z parts (whole 512-voxel multiples; the last part takes the rest) ---
occlv() {   # --occupancy L for the fields: the first of levels LV+3, LV+2, LV+4 the CT has (local root or mirror URL); none: no flag
  local l; for l in $((LV + 3)) $((LV + 2)) $((LV + 4)); do
    case $R in http://*|https://*) curl -sfI $R/$K/$l/zarr.json > /dev/null && { echo "--occupancy $l"; return; };;
                     *) [ -e $R/$K/$l/zarr.json ] && { echo "--occupancy $l"; return; };; esac
  done
}
# FCTR / FCTK (default the run's CT): another CT root / key for the fields predict only, e.g. an unmasked or dilated-mask scan (audit-ugeom
# 2026-11-14: 16.2 % of the published 1667 surface lies on zero (masked) CT, the outer turns clipped at the mask edge)
# FBLEND=1 (OFF by default again since 2026-11-17: the wbench GT A/B, 44 boxes, 1 grid ids 0.533 vs 2 grids 0.531, 4 0.524, 8 0.517 -- the blend removes the seam steps (audit #5) but does not move ids; FBLEND=0 the single grid, ~2x faster fields + the streamed ref): each part predicted on two tile grids (origin 0 and half the stride) and
# blended with ufsm field-blend --tile (sin^2 weights: every cell leans on the window it sits deep inside), against the tile-border steps of
# the single-grid fields (1667 u1 640/64: the normal plane t0 steps 4-5x its interior median at every 256-cell border). Lat fields only.
# A/B 2026-11-16 (1667 u1 13000, 512 x 2048^2 level-2 box, scripts/unroll/seam_ab.sh): 256-border step / interior median, single grid vs
# blended: phase y 1.43 -> 1.11, x 1.78 -> 1.09; t0 y 3.95 -> 2.42, x 6.08 -> 3.08. 896/192 windows: single 1.36 / 1.60 / 3.60 / 5.61, blended
# the same as 640/64 blended at 2x the predict time: the blend, not the halo, removes the seams. 1152 windows do not fit a 16 GB card.
# ONEPASS=1 (OFF by default; infer audit #8, A/B after the mega pass): the fit fields AND the ink heads (ink, c12, c13) from ONE predict
# per z part: window 1152 / halo 64, --seam-inproc 1 (the heads blended over two tile grids on the GPU, the fields from grid 0),
# heads full-grid into $O/ink/onepass (the parts' shards coexist); the ink stage then only links ink / c12 / c13. ~40 % less GPU time
# (estimate: 2.85x computed voxels per occupied voxel against ~4.5-5.9x). Needs a predict with --seam-inproc + --fit-zarr (inferfix).
onepass_on() { [ "${ONEPASS:-0}" = 1 ] && [ "$FKIND" = lat ] && [ "$INKMODE" = lat ]; }
fpred() {   # fpred GPU BOX OUT
  if onepass_on; then
    X $O/logs/fields.log $U predict $CK $R $K $O/ink/onepass --um $UM --level $LV --box $2 --full-grid 1 --shard 512 --window ${OPWIN:-1152} --halo 64 \
      --seam-inproc 1 --heads ink,c12,c13 --fit-zarr $3 --fit-h 2 --fit-q 8 --fit-q-phase 4 --axis $O/axis.json $(occlv) --q 8 --levels 4 \
      --threads ${PTH:-14} --gpu $(lg $1) --ema 1 && { [ "$DRY" = 1 ] || [ -e $3/fields.json ]; }; return
  fi
  if [ "${FBLEND:-0}" = 1 ] && [ "$FKIND" = lat ]; then
    local st=$(( ${LWIN:-640} - 2 * ${LHALO:-64} )) gg
    for gg in 0 $((st / 2)); do fpred1 $1 $2 $3.g$gg $gg || return 1; done
    X $O/logs/fields.log $U field-blend $3 $3.g0 $3.g$((st / 2)) --tile $st,$st,$st --grids 0,0,0:$((st / 2)),$((st / 2)),$((st / 2)) --threads $TH || return 1
    [ "$DRY" = 1 ] || { [ -e $3/fields.json ] && rm -rf $3.g0 $3.g$((st / 2)); }; return
  fi
  fpred1 $1 $2 $3 0
}
fpred1() {   # fpred1 GPU BOX OUT GRID: one predict of the fit fields, its windows anchored at GRID (each axis)
  # a box off the window grid runs each window once per 512 shard it touches (predict.c: up to 8x on the half-stride grid; 0332 FBLEND
  # 3138 windows against 505 on grid 0, 34 of its 46 min fields): the box is started back on the grid instead (field-blend crops it to the
  # grid-0 cells; the blended fields byte-identical, DESIGN 2026-11-16 unroll perf)
  local box=$2 st=$(( ${LWIN:-640} - 2 * ${LHALO:-64} )) b d; IFS=, read -r -a b <<< "$2"
  if [ "$4" != 0 ] && [ "$FKIND" = lat ]; then for d in 0 1 2; do
    local s=$(( ((b[d] - $4) % st + st) % st )); b[d]=$(( b[d] - s )); b[d + 3]=$(( b[d + 3] + s )); done   # (before 0 at a box at 0: predict reads fill there)
    box=$(IFS=,; echo "${b[*]}"); fi
  # FSKIP (lat fields; default 20, 0 = dense): --skip-empty 1 --skip-thr FSKIP --skip-margin 1, void tiles (no CT voxel above FSKIP) left at
  # their bias. wbench GT A/B 2026-11-17 (u1 13000, 44 boxes): ids dense 0.5327, thr 20 0.5369 (+0.004 [+0.0003, +0.009]), 40 0.5357
  # (+0.003, CI > 0), 60 0.5294 (n.s.); the gain sits on the packed Paris 4 boxes (+0.015 / +0.005); fields predict -13 % on 0332 / 1667
  local win sk=; [ "$FKIND" = lat ] && [ "${FSKIP:-20}" != 0 ] && sk="--skip-empty 1 --skip-thr ${FSKIP:-20} --skip-margin 1"
  if [ "$FKIND" = lat ]; then win="--window ${LWIN:-640} --halo ${LHALO:-64}"
  else win="$([ "${FWIN:-prod}" = auto ] && echo "--window auto" || echo "--window-z 384 --halo-z 64 --window 640") --halo 64"; fi
  X $O/logs/fields.log $U predict $CK ${FCTR:-$R} ${FCTK:-$K} $3.tmp --um $UM --level $LV --box $box $win --shard 512 \
    --grid-origin $4,$4,$4 --ema 1 --q 8 --threads ${PTH:-14} --gpu $(lg $1) --axis $O/axis.json --levels 1 --fit-zarr $3 --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 $sk $(occlv) &&
    { [ "$DRY" = 1 ] || { rm -rf $3.tmp; [ -e $3/fields.json ]; }; }
}
# --- the ref's pass 1 streamed behind the fields predict (REFSTREAM; 6a, 11-12) ---
# Each z part's predict appends "z0 z1" (part cells) to <part>/rows.done per flushed field slab. refstream links the finished shard rows
# of every part into one view (OUT: zarr.json shapes and fields.json of the whole box, the parts' c/<z> rows linked at their box offsets,
# the box rows in OUT/rows.ready) and runs count-field --pool-only --follow on it: the chunks in the parts' order, each once its rows are
# written. The pooled sums equal a whole-box pass 1 (per chunk, disjoint solve rows; the prior from the middle row's chunk; the chunk
# rows fixed by refcz in both runs), so the ref (--pool-cache) is byte-identical to the stage-barrier ref with those chunk rows.
refstream_on() { [ "${REFSTREAM:-1}" = 1 ] && [ "${FBLEND:-0}" != 1 ] && ! onepass_on && [ "$FKIND" = lat ] && [ "${RECTOSP:-1}" = 0 ] && [ "${CTCAL:-0}" != 1 ] && [ -z "${REFCMD:-}" ] && [ "$DRY" != 1 ]; }
refstream() {   # refstream OUT PART..
  local S=$1; shift; local parts=("$@") k p pl z a b n=0 starts=() fs cz cfp=
  rm -rf $S $S.fin $O/ref.pool $O/ref.pool.ok; mkdir -p $S
  for p in "${parts[@]}"; do while [ ! -s $p/fields.partial.json ] || [ ! -e $p/pc/zarr.json ]; do [ -e $S.fin ] && return 1; sleep 2; done; done
  for p in "${parts[@]}"; do starts+=($n); n=$(( n + $(jq -r '.shape[0]' $p/fields.partial.json) )); done
  fs=$(jq -r '.chunk_grid.configuration.chunk_shape[0]' ${parts[0]}/pc/zarr.json)
  for k in $(seq 1 $((${#parts[@]} - 1))); do [ $(( ${starts[$k]} % fs )) = 0 ] || { log "refstream: part $k starts at cell ${starts[$k]}, not a multiple of the shard $fs"; return 1; }; done
  for pl in $(ls ${parts[0]}); do [ -e ${parts[0]}/$pl/zarr.json ] || continue; mkdir -p $S/$pl/c; jq -c --argjson n $n '.shape[0] = $n' ${parts[0]}/$pl/zarr.json > $S/$pl/zarr.json; done
  # fields.json: the first part's, shape and axis over all parts (fields_cat.sh's rule: part k's rows 0 .. Zk h - 1, the last part's all)
  local h=$(jq -r .h ${parts[0]}/fields.partial.json) j=$(jq -c . ${parts[0]}/fields.partial.json)
  for k in $(seq 1 $((${#parts[@]} - 1))); do j=$(echo "$j" | jq -c --argjson k $(( ${starts[$k]} * h )) --slurpfile b ${parts[$k]}/fields.partial.json '.axis = (.axis[0:$k] + $b[0].axis)'); done
  echo "$j" | jq -c --argjson n $n '.shape[0] = $n' > $S/fields.json; : > $S/rows.ready
  cz=$(refcz $S/fields.json)
  ( env $(sgpu) nice -n 10 $U count-field $S $O/cf.unused $(refpool) --pool-cache $O/ref.pool --pool-only 1 --model $O/ref.pool.unused --chunk-rows $cz --follow $S/rows.ready \
      --follow-split ${starts[1]:-0} --threads ${REFSTH:-6} --read-threads ${REFSTH:-6} --mem-gb ${CFMEM:-24} >> $O/logs/refstream.log 2>&1 && echo $cz > $O/ref.pool.ok ) & cfp=$!
  local seen=() fin=0; for k in "${!parts[@]}"; do seen+=(0); done
  while :; do
    [ -e $S.fin ] && fin=1
    for k in "${!parts[@]}"; do p=${parts[$k]}; local L; L=$(wc -l < $p/rows.done 2> /dev/null || echo 0)
      while [ ${seen[$k]} -lt $L ]; do seen[$k]=$(( ${seen[$k]} + 1 )); read -r a b < <(sed -n "${seen[$k]}p" $p/rows.done)
        for z in $(seq $((a / fs)) $(( (b - 1) / fs ))); do for pl in $(ls $S); do [ -d $p/$pl/c/$z ] && ln -sfn $p/$pl/c/$z $S/$pl/c/$(( z + ${starts[$k]} / fs )); done; done
        echo "$(( a + ${starts[$k]} )) $(( b + ${starts[$k]} ))" >> $S/rows.ready; done; done
    [ $fin = 1 ] && break; sleep 5
  done
  # (the predicts are done: every row is linked; the follower finishes its last chunks, or fails: no ref.pool.ok, the ref pools itself)
  if ! wait $cfp; then rm -f $O/ref.pool $O/ref.pool.ok; return 1; fi
  log "refstream: pass 1 pooled behind the predicts (chunk rows $cz): $(grep -E 'pass 1' $O/logs/refstream.log | tail -n 1)"
}
fields() {
  local box; box=$([ "$DRY" = 1 ] && echo ${BOX:-BOX} || cat $O/box.txt); local g=($FIELDGPUS) n=${#g[@]}
  [ "$DRY" = 1 ] || { rm -rf $F; mkdir -p $O/fields; rm -f $O/ref.pool $O/ref.pool.ok; }
  [ "$DRY" = 1 ] && [ "${REFSTREAM:-1}" = 1 ] && [ "$FKIND" = lat ] && [ "${RECTOSP:-1}" = 0 ] && echo "    (REFSTREAM: the ref's pass 1 behind the predicts: count-field --pool-only --follow on the parts' finished rows -> ref.pool)"
  local z0 dz y0 x0 dy dx k a b pids=() parts=(); IFS=, read -r z0 y0 x0 dz dy dx <<< "$box"
  [ $(( (dz / n) / 512 )) -lt 1 ] && n=1   # (parts of >= 512 voxels each)
  if [ $n = 1 ]; then fpred ${g[0]} $box $F; return; fi
  # z parts: equal shares rounded to the NEAREST 512 voxels (= 256 field cells at h 2, a multiple of the 128-cell field shard, of the 512
  # latent stride and the 256 voxel z stride; box origins are multiples of 512). Was floor to 1024: 1667 split 4096 / 5173 (69 / 89 s)
  for k in $(seq 0 $((n - 1))); do
    a=$(( z0 + (dz * k / n + 256) / 512 * 512 )); b=$(( k == n - 1 ? z0 + dz : z0 + (dz * (k + 1) / n + 256) / 512 * 512 ))
    parts+=($F.p$k); [ "$DRY" = 1 ] || rm -rf $F.p$k $F.p$k.tmp
    fpred ${g[$k]} $a,$y0,$x0,$((b - a)),$dy,$dx $F.p$k & pids+=($!)
  done
  # REFSTREAM (default 1 for latent fields without rectosp / ctcal / REFCMD): the ref's pass 1 (pooling) runs behind the predicts
  local sp=; if refstream_on; then refstream $F.stream "${parts[@]}" & sp=$!; fi
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ -n "$sp" ] && { touch $F.stream.fin; wait $sp || log "fields: the streamed ref pass 1 failed (the ref runs it itself)"; rm -rf $F.stream $F.stream.fin; }
  [ $rc = 0 ] || return 1
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
# REFPH: the ref's phase-target flags. Default (COUNT 11-01): confident phase gradients at weight 0.5, x16 where the network spacing is wide
# (ramp 100..140 um): the phase counts right on wide sheets and skips packed ones, where the density counts. Gate vs the old --phase-w 4
# (whole-scroll refs, per-z-slab GT): P4 ids all 6 slabs up (z2-z7 .209/.213/.181/.151/.260/.239 -> .217/.226/.202/.184/.283/.254),
# 0.5 mm pairs all 6 up (mean .679 -> .695), 4 mm pairs mean .286 -> .293, GT
# intervals off .099 -> .083; 1667 all 5 slabs up (ids +.06..+.13, 4 mm pairs +.06..+.08); 0139 flat; 0846A wraps / CT peaks per turn same.
# REFPH="--phase-w 4" = the old ref.
# refpool: the ref's pass-1 (pooling) flags. (refcz was: the chunk rows count-field picks from --mem-gb CFMEM when memory is not
# short, 12 B per cell of a cross-section, a multiple of the pool 8, then of 128, until 2026-11-17)
# MSTRIDE 1 (was 4): the model's local gauge from every chunk. With 4 only every 4th chunk had one (count-field now interpolates the rows
# between, but the 1667 refs were coherent on 1 / K of z: unroll #8 recheck 11-14); the full pass costs ~4x a 7 s GPU model pass
refpool() { echo "--R 0 --pool 8 ${REFPH---phase-w 0.5 --wide 100,140 --wide-phase 15} --model-only 1 --model-stride ${MSTRIDE:-1} ${REFX:-}"; }
# refcz: the ref's chunk rows, fixed (REFCZ, 512; lead 2026-11-17): they had come from the free host memory (--mem-gb lowered to 60 % of
# what was available) and change the ref (0332: 512 vs 640 rows, delta p99 0.017 turn, max 1.03, so two runs' joined models differed).
# count-field takes fewer only when --mem-gb cannot hold them, with a DEFAULTED: line; the ref stage logs the rows it used
refcz() { echo ${REFCZ:-512}; }
# SPIRALDIR: the spiral's hand for the ref (count-field --dir; the slabs take it from the ref). Default: scripts/unroll/spiral_dir.tsv
# (measured against published surfaces), else count-field's own vote. Its ring vote can be empty (PHerc1667: sheets far from circles),
# and then it fell back to +1 silently: the 11-12 1667 unroll had the wrong hand (sheet identity vs the published unwrap 0.48; the ref
# with --dir -1: 0.73, DESIGN 2026-11-14). An empty vote is logged to progress.log.
# spiral_dir.tsv: scroll, dir (+1 / -1; anything else = unknown), source (GT | published | estimator | unknown), evidence. Without an
# entry count-field decides itself and REFUSES when it has no evidence (no ring vote): set SPIRALDIR, add the scroll, or REFX="--dir-default-ok 1".
spiraldir() { [ -n "${SPIRALDIR:-}" ] && { echo $SPIRALDIR; return; }; awk -F'\t' -v s=$S '$1 == s && ($2 == "1" || $2 == "+1" || $2 == "-1") { print $2 }' $H/spiral_dir.tsv 2> /dev/null | head -1; }
ref() {
  [ "$DRY" = 1 ] || rm -rf $REF
  local sd sds; sd=$(spiraldir); [ -n "$sd" ] && REFX="${REFX:-} --dir $sd"
  sds=$([ -n "${SPIRALDIR:-}" ] && echo env || { [ -n "$sd" ] && echo tsv || echo gate; })
  local cal=; [ "${CTCAL:-0}" = 1 ] && cal="--ct-cal $O/ct.cal"
  # the pass-1 sums streamed behind the fields predict (fields(): REFSTREAM): the same sums, so only the solve and model pass run here
  local pc=; [ "$DRY" != 1 ] && [ -e $O/ref.pool.ok ] && pc="--pool-cache $O/ref.pool --chunk-rows $(cat $O/ref.pool.ok)"
  # DIRCOH=1 (ON by default since 2026-11-16; DIRCOH=0 refuses an unknown hand outright): measured (audit unroll #1, model-only solves): 1667
  # -1 0.1731 vs +1 0.0986 (the published hand, margin 0.075); Paris 4 +1 0.0072 vs -1 0.0062 (margin 0.001 < 0.05: REFUSED, never wrong):
  # where it cannot tell it refuses as before, so on is never worse. audit-unroll 3a: no env / tsv hand -> two LS solves, --dir +1 into REF.p and -1 into REF.m, each
  # printing "count-field: model coherence C" (|sum z| / sum |z| of the phase-vs-model phasors before smoothing; the wrong hand turns them
  # twice per revolution, so C ~ 0). The higher C wins by >= DIRCOH_MARGIN (0.05), else refuse. The pooled sums are dir-free (--pool-cache reused).
  if [ -z "$sd" ] && [ "${DIRCOH:-1}" = 1 ] && [ -z "${REFCMD:-}" ]; then
    local d c cp= cm=
    for d in p m; do
      [ "$DRY" = 1 ] || rm -rf $REF.$d
      X $O/logs/ref.$d.log $U count-field $F $O/cf.unused $(refpool) --dir $([ $d = p ] && echo 1 || echo -1) --iters 1200 --model $REF.$d \
        --threads $TH --mem-gb ${CFMEM:-24} $cal ${pc:---chunk-rows $(refcz)} || { log "ref: DIRCOH: the --dir $d solve failed (logs/ref.$d.log)"; return 1; }
      [ "$DRY" = 1 ] && continue
      c=$(grep -oE 'count-field: model coherence [0-9.]+' $O/logs/ref.$d.log | tail -1 | awk '{print $4}')
      [ -n "$c" ] && [ -e $REF.$d/model.json ] || { log "ref: DIRCOH: no model coherence / model from the --dir $d solve"; return 1; }
      [ $d = p ] && cp=$c || cm=$c
    done
    [ "$DRY" = 1 ] && return 0
    local win; win=$(awk -v p=$cp -v m=$cm -v g=${DIRCOH_MARGIN:-0.05} 'BEGIN{print p - m >= g ? "p" : m - p >= g ? "m" : ""}')
    log "ref: DIRCOH: model coherence +1 $cp, -1 $cm (margin ${DIRCOH_MARGIN:-0.05}) -> ${win:-REFUSED}"
    [ -n "$win" ] || { log "ref: REFUSED: DIRCOH could not tell the hand; set SPIRALDIR or scripts/unroll/spiral_dir.tsv"; return 1; }
    mv $REF.$win $REF && cp $O/logs/ref.$win.log $O/logs/ref.log && rm -rf $REF.p $REF.m
    if type mset > /dev/null 2>&1; then mset spiral_dir "$(jq -r .dir $REF/model.json 2> /dev/null)"; mset spiral_dir_src "gate:coherence +1 $cp -1 $cm"; fi
    return 0
  fi
  if [ -n "${REFCMD:-}" ]; then X $O/logs/ref.log $REFCMD $F $REF $cal; else
  X $O/logs/ref.log $U count-field $F $O/cf.unused $(refpool) --iters 1200 --model $REF \
    --threads $TH --mem-gb ${CFMEM:-24} $cal ${pc:---chunk-rows $(refcz)}; fi && { [ "$DRY" = 1 ] || [ -e $REF/model.json ]; } ||
    { grep -q 'REFUSED' $O/logs/ref.log 2> /dev/null && log "ref: $(grep -o 'REFUSED.*' $O/logs/ref.log | tail -1 | cut -c1-220)"; return 1; }
  [ "$DRY" != 1 ] && log "ref: $(grep -oE 'count-field: dir [+-]1 \([^)]*\)' $O/logs/ref.log | tail -1)"
  [ "$DRY" != 1 ] && log "ref: chunk rows $(grep -hoE '[0-9]+ rows per chunk' $O/logs/ref.log $O/logs/refstream.log 2> /dev/null | tail -1 | awk '{print $1}') (REFCZ ${REFCZ:-512})$(grep -hq '^DEFAULTED: count-field chunk rows' $O/logs/ref.log $O/logs/refstream.log 2> /dev/null && echo ': DEFAULTED, see logs/ref.log')"
  # the run manifest (audit-unroll): the direction the ref used and where it came from (env / tsv / gate = count-field's own evidence)
  if [ "$DRY" != 1 ] && type mset > /dev/null 2>&1; then
    mset spiral_dir "$(jq -r .dir $REF/model.json 2> /dev/null)"; mset spiral_dir_src "$sds$([ $sds = gate ] && grep -oE 'count-field: dir [+-]1 \(([^)]*)\)' $O/logs/ref.log | tail -1 | sed -E 's/.*\((.*)\)/: \1/')"; fi
  [ "$DRY" != 1 ] && grep -q 'no ring vote on the direction' $O/logs/ref.log && log "ref: WARNING: $(grep -o 'no ring vote on the direction.*' $O/logs/ref.log | tail -1)"
  [ "$DRY" != 1 ] && grep -q 'WARNING no-data pairs' $O/logs/ref.log && log "ref: ALARM $(grep -o 'no-data: .*' $O/logs/ref.log | tail -1)"
  return 0
}

# --- rectosp: m7 recto spacing into the count target (see the header); the source: RECTOSRC, else the mirror's
#     <SCROLL>/representations/predictions/surfaces/<scan id>-surface-*-surface-m7-L<LV>-prob.zarr at its finest level ---
m7src() {
  [ -n "${RECTOSRC:-}" ] && { echo "$RECTOSRC"; return; }
  local base=https://dl.ash2txt.org/community-uploads/forrest/volcomp/$S/representations/predictions/surfaces id z key
  id=$(basename $K | cut -c1-14)
  z=$(curl -sf --max-time 60 $base/ | grep -o "href=\"$id-surface-[^\"]*-surface-m7-L$LV-prob.zarr/\"" | head -1 | sed 's/href="//; s/\/"$//')
  [ -n "$z" ] || return 1
  key=$(curl -sf --max-time 60 $base/$z/ | grep -o 'href="[0-9.]*/"' | sed 's/href="//; s/\/"$//' | sort -g | head -1)
  [ -n "$key" ] && echo "$base/$z $key $key"
}
rectosp() {
  local src; src=$(m7src) || { log "rectosp: no m7 prob for $S $(basename $K | cut -c1-14) level $LV on the mirror: skipped (fields unchanged)"; [ "$DRY" = 1 ] || echo none > $O/rectosp.src; return 0; }
  log "rectosp: source $src"; [ "$DRY" = 1 ] || echo "$src" > $O/rectosp.src
  [ "$DRY" = 1 ] || rm -rf $O/fields/rsp
  X $O/logs/rectosp.log $U recto-sp $F $O/fields/rsp --src $src --cache $O/fields/rsp.cache --threads $TH ${RECTOSPX:---half-um 400 --half2-um 800 --blend 0.3} &&
    { [ "$DRY" = 1 ] || { rm -rf $O/fields/rsp.cache; [ -e $O/fields/rsp/sp/zarr.json ]; }; }
}

# --- ink (another agent's recipe via INKCMD SCROLLDIR GPU OUT; default the feathered single pass of scripts/corpus/inklib.sh) ---
# inkcal: the frozen GroupNorm statistics (gn.bin) from the main pass's windows (INKW / INKH: auto / 32; was 640 / 64 against the
# main pass's auto = 576 / 32, audit 2026-11-14 #19) of a 2048-row band at the box's middle; its own stage so it can
# run on GPU INKGPU beside the ref (INKAT=slabs) while the main pass waits for the slabs. INKCMD: nothing (the recipe does its own).
# INKCALZ: the band's rows (2048; P4: 505 windows, 148 s)
inkcal() {
  { [ -n "${INKCMD:-}" ] || onepass_on; } && return 0
  [ "$DRY" = 1 ] || mkdir -p $O/ink
  local box cal; box=$([ "$DRY" = 1 ] && echo ${BOX:-BOX} || cat $O/box.txt)
  cal=$(echo $box | awk -F, -v cz=${INKCALZ:-2048} '{ zc = $1 + int($4 / 2); z0 = zc - int(cz / 2); if (z0 < $1) z0 = $1; nz = cz; if (nz > $4) nz = $4; printf "%d,%d,%d,%d,%d,%d", z0, $2, $3, nz, $5, $6 }')
  local P="--um $UM --level $LV --shard 512 --ema 1 --q 32 --threads ${ITH:-12} --gpu $(lg $INKGPU) --axis $O/axis.json $(occlv)"
  [ -s $O/ink/gn.bin ] || { X $O/logs/inkcal.log $U predict $INKCK $R $K $O/ink/gncal.zarr $P ${INKPX:-} --window ${INKW:-auto} --halo ${INKH:-32} --box $cal --full-grid 1 --levels 1 --no-out 1 --gn-save $O/ink/gn.tmp &&
                             { [ "$DRY" = 1 ] || { mv $O/ink/gn.tmp $O/ink/gn.bin; rm -rf $O/ink/gncal.zarr; }; } || return 1; }
}
ink() {
  [ "$DRY" = 1 ] || mkdir -p $O/ink
  # INKCMD set but empty (a pod env with an unset path, "INKCMD=$X"): fail loudly instead of falling back to the default prediction;
  # any INKCMD must leave a readable zarr at $INK (a dangling link from a missing published ink fails here, not in the renders)
  if [ -n "${INKCMD+x}" ] && [ -z "${INKCMD// /}" ]; then log "ink: INKCMD is set but EMPTY: refusing (unset it for the default feathered ink)"; return 1; fi
  if onepass_on; then   # (ONEPASS: the heads came with the fields) link them where the strip and its gate read them
    [ "$DRY" = 1 ] && { echo "    (ONEPASS: link $O/ink/onepass/{ink,c12,c13})"; return 0; }
    local h; for h in ink c12 c13; do [ -e $O/ink/onepass/$h/zarr.json ] || { log "ink: ONEPASS: no $h in $O/ink/onepass"; return 1; }; done
    rm -rf $INK $O/ink/c12.zarr $O/ink/c13.zarr; ln -s onepass/ink $INK; ln -s onepass/c12 $O/ink/c12.zarr; ln -s onepass/c13 $O/ink/c13.zarr; return 0
  fi
  if [ -n "${INKCMD:-}" ]; then UNROLL_CT="$R|$K|$UM|$LV" X $O/logs/ink.log $INKCMD $O $(lg $INKGPU) $INK || { log "ink: INKCMD '$INKCMD' failed (logs/ink.log)"; return 1; }
    [ "$DRY" = 1 ] || [ -e $INK/zarr.json ] || { log "ink: INKCMD '$INKCMD' left no zarr at $INK (zarr.json missing; dangling link?)"; return 1; }
    return 0; fi
  local box cal; box=$([ "$DRY" = 1 ] && echo ${BOX:-BOX} || cat $O/box.txt)
  cal=$(echo $box | awk -F, '{ zc = $1 + int($4 / 2); z0 = zc - 1024; if (z0 < $1) z0 = $1; nz = 2048; if (nz > $4) nz = $4; printf "%d,%d,%d,%d,%d,%d", z0, $2, $3, nz, $5, $6 }')
  local P="--um $UM --level $LV --shard 512 --ema 1 --q 32 --threads ${ITH:-12} --gpu $(lg $INKGPU) --axis $O/axis.json $(occlv)"
  inkcal || return 1
  # one feathered pass (inklib.sh): window auto (576/32 on 16 GB, 1088/32 from ~46 GB free), each border blended over 32 voxels; resumable
  X $O/logs/ink.log $U predict $INKCK $R $K $INK $P ${INKPX:-} --box $box --window ${INKW:-auto} --halo ${INKH:-32} --feather ${INKF:-32} --full-grid 1 --resume 1 --levels 4 \
    --gn-frozen $O/ink/gn.bin || return 1
}

# --- slab fits: one lane per GPU, claims by flock (fd.sh lane) ---
# SLABFIT: the slab fits' step counts and minibatch: stages 4 / 5 (the fine lattices) 200 Adam steps instead of 300 and 1 M sample
# points instead of 2 M (10-31 gate on 0490B + P4 crops: fit -22% (min of 3 interleaved), wrap-check same, join agreement .762 ->
# .839 / .937 -> .960, P4 GT ids -.0006, pairs -.0004..-.0015 = within the fieldq q4 drift); SLABFIT="" = fit's defaults
SLABFIT=${SLABFIT---steps 300,300,300,300,200,200 --batch 1000000}
# CROPM: fit --crop-material CROPM (cells of margin around the slab's material y / x bounds; 0 = off): the field arrays over the material only, the
# model byte-identical to the whole-box fit (positions, lattice, minibatch hash stay the box's; slabperf-20261112: CPU backend on P4 boxes,
# test_fit). P4 slabs keep ~0.45-0.6 of the 3575 x 3712 cross-section, so their fields (12.7 GB) fit the 16 GB device instead of managed memory.
# FITLOCK=1: UFSM_FIT_GPU_LOCK per GPU (one fit on the device at a time, from its host planes to its exit), for FITLANES=2 on one card: the
# next slab's material box / host-plane decode overlaps the current one's device stages; FITLOCK=2: also UFSM_FIT_LOCK_LATE=1 (the normals decoded
# to the host before the lock too: +6 B per cell of host memory per lane while loading)
CROPM=${CROPM-2}; FITLOCK=${FITLOCK:-0}
lane() {   # lane GPU
  local g=$1 k s d c0 c1 a b t0 rc ze ny nx dir
  if [ "$DRY" = 1 ]; then echo "    lane GPU $g: per slab k: mkcrop.sh $F slabs/sNNN/f ($I cells + $OV overlap); $U fit slabs/sNNN/f --gpu $(lg $g) --out slabs/sNNN/fit --dir REFDIR --ref-model $REF --ref-snap 1 --uw-repair 0 $([ "${CROPM:-0}" != 0 ] && echo --crop-material $CROPM) $SLABFIT [phase gate on by default; FITX=\"--phase-gate 0\" off] ${FITX:-}"; return 0; fi
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
    env UFSM_FIT_THREADS=$FTH_LANE $([ "$man" = 1 ] && echo UFSM_FIT_MANAGED=1) $([ "$FITLOCK" != 0 ] && echo UFSM_FIT_GPU_LOCK=$O/state/fitgpu$g.lock) $([ "$FITLOCK" = 2 ] && echo UFSM_FIT_LOCK_LATE=1) /usr/bin/time -v nice -n 5 \
      $U fit $d/f --gpu $(lg $g) --out $d/fit --dir $dir --ref-model $REF --ref-snap 1 --uw-repair 0 $([ "${CROPM:-0}" != 0 ] && echo --crop-material $CROPM) $SLABFIT ${FITX:-} > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s$([ "$man" = 1 ] && echo " managed")" >> $O/slabs.txt
    if [ $rc = 0 ]; then touch $d/done; elif grep -q "no data for the prior" $d/fit.log; then touch $d/empty $d/done; fi
    rm -rf $d/f; exec 9>&-
  done
}
# fitlanes: the slab-fit lanes, a GPU id per lane. Each GPU of FITGPUS gets FITLANES lanes; FITLANES unset means auto: the device's
# free GB (ufsm sysinfo, at the stage's start) - 1 over a fit's estimated device memory, 12 B per slab cell ((I + 2 OV) x ny x nx of the
# fields) + 0.5 GB, clamped to 1..3 (2026-11-17: 0332's slabs are 96 x 1392 x 1504 = 0.2 G cells, ~1.9 GB per fit measured: 3 lanes on
# one 16 GB 5060 Ti 232 -> 132 s for the 62 slabs, 4 lanes 134 s (95 % GPU busy), every delta.f32 byte-identical; a Paris 4 slab, ~1.3 G
# cells, ~13 GB: 1 lane on 16 GB cards, 3 on 96 GB). Without the fields' shape (DRY): (device GB - 8) / 16, as before.
# Measured on a 96 GB RTX PRO 6000 with 6 Paris 3 slabs and threads = CPUs / lanes: 1 lane 82.8 s, 2 lanes 54.8 s,
# 3 lanes 47.9 s, delta.f32 byte-identical. A fit takes ~10-16 GB of device memory. The device size comes from
# `ufsm sysinfo` (CUDA order, MIG-aware); when it is unknown, 1 lane.
fitlanes() {
  local g n gb fr i out= ny nx est=0
  ny=$(jq -r '.shape[1]' $F/fields.json 2> /dev/null); nx=$(jq -r '.shape[2]' $F/fields.json 2> /dev/null)
  case "$ny$nx" in *[!0-9]*|'') ;; *) est=$(awk -v a=$((I + 2 * OV)) -v y=$ny -v x=$nx 'BEGIN { printf "%d", 12 * a * y * x / 1e8 + 5 }');; esac   # (0.1 GB)
  for g in $FITGPUS; do
    n=${FITLANES:-}
    if [ -z "$n" ] && [ $est -gt 0 ]; then fr=$($U sysinfo 2> /dev/null | awk -v g=$(lg $g) '$1 == "gpu" && $2 == g ":" && match($0, /\([0-9.]+ GB free\)/) { print int(10 * substr($0, RSTART + 1, RLENGTH - 10)) }')
      n=$(( (${fr:-0} - 10) / est )); [ $n -lt 1 ] && n=1; [ $n -gt 3 ] && n=3
    elif [ -z "$n" ]; then gb=$($U sysinfo 2> /dev/null | awk -v g=$(lg $g) '$1 == "gpu" && $2 == g ":" && match($0, /, [0-9.]+ GB \(/) { print int(substr($0, RSTART + 2, RLENGTH - 6)) }')
      n=$(( (${gb:-0} - 8) / 16 )); [ $n -lt 1 ] && n=1; [ $n -gt 3 ] && n=3; fi
    for i in $(seq $n); do out+="$g "; done
  done; echo $out
}
slabs() {
  local g pids=() ng lanes
  # DEMOD=1 (off by default; audit ugeom #2: the slab fits change no integers on the 1667 / P4 GT boxes, only the sub-turn): no slab
  # fits; join takes the ref's integers and the phase's sub-turn (ufsm cf-demod --out, DEMODR nodes)
  [ "${DEMOD:-0}" = 1 ] && { log "slabs: skipped (DEMOD=1: join = cf-demod of the ref)"; return 0; }
  # fit threads per lane: the host's CPUs over the lanes (fits are thread-count independent: 0490B s020/s021 models equal at 10 / 20;
  # 1 lane 10 -> 20 threads: 8 slabs 163 -> 150 s; 2 lanes per 16 GB GPU with the GPU lock: 156 s, so 1 lane there; 96 GB: 3, fitlanes)
  lanes=$(fitlanes); [ "$DRY" = 1 ] && echo "    slab lanes (GPU per lane): $lanes"
  ng=$(echo $lanes | wc -w); FTH_LANE=${FTHREADS:-$(( $(ncpu) / (ng > 0 ? ng : 1) ))}; [ $FTH_LANE -lt 2 ] && FTH_LANE=2
  for g in $lanes; do
    if [ "$g" = "$INKGPU" ] && [ "${NOINK:-0}" != 1 ] && [ "$INKAT" = fields ] && [ "${INKWAIT:-1}" = 1 ] && ! done_ ink; then { [ "$DRY" = 1 ] && echo "    (GPU $g lane starts once the ink is done or failed)" || while ! done_ ink && ! stopped && [ ! -e $O/state/ink.failed ]; do sleep 30; done; lane $g; } &
    elif [ "$g" = "$INKGPU" ] && [ "${NOINK:-0}" != 1 ] && [ "$INKAT" != fields ] && ! done_ inkcal; then { [ "$DRY" = 1 ] && echo "    (GPU $g lane starts once the ink calibration is done or failed)" || while ! done_ inkcal && ! stopped && [ ! -e $O/state/inkcal.failed ]; do sleep 10; done; lane $g; } &
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
  if [ "${DEMOD:-0}" = 1 ]; then [ "$DRY" = 1 ] || rm -rf $O/joined
    X $O/logs/join.log $U cf-demod $REF $F --out $O/joined --R ${DEMODR:-1} --L ${DEMODL:-8} --step ${DEMODSTEP:-2} --threads $TH || return 1
    [ "$DRY" = 1 ] || log "join: DEMOD=1: $(grep '^cf-demod: demodulated' $O/logs/join.log | tail -1)"; return 0; fi
  if [ "$DRY" = 1 ]; then X $O/logs/join.log $U fit-join $O/joined $F --blend 16 --ref $REF "slabs/sNNN/fit 2c0 2c1 ..."; return; fi
  bad=$(badslabs); [ -z "$bad" ] || { log "join: REFUSED: bad slab fits $(for d in $bad; do basename $d; done | tr '\n' ' ')(rm state/slabs.done to refit)"; return 1; }
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; [ -e $d/empty ] && continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; X $O/logs/join.log $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}"
}
# check: wrap-check of the joined model (CT papyrus peaks between adjacent wraps along radial rays; per-wrap verdicts real / dup / merged
# in wrapcheck.tsv). --ct-um is the CT level's um ($LUM); 0 CT peaks per turn overall = a wrong level / um: the stage fails
check() {
  X $O/logs/check.log $U wrap-check $O/joined $F --ct $R $K/$LV --ct-um $LUM --tsv $O/wrapcheck.tsv.tmp --threads $TH || return 1
  [ "$DRY" = 1 ] && { echo "    $U unroll-qa --fields $F --model $O/joined --ct $R $K/$LV --out $O/qa/unroll-qa --tiles ${UQTILES:-48}${FIBRES:+ --fibres $FIBRES}$([ "${UQSTRICT:-0}" = 1 ] && echo ' --strict')  > $O/logs/unrollqa.log"; return 0; }
  [ -s $O/wrapcheck.tsv.tmp ] && [ $(awk -F'\t' 'NR > 1 && $NF != ""' $O/wrapcheck.tsv.tmp | wc -l) -gt 0 ] || { log "check: wrap-check wrote no verdicts"; return 1; }
  awk -F'\t' 'NR > 1 { n++; e += $3 } END { exit !(n > 0 && e / n < 0.999) }' $O/wrapcheck.tsv.tmp ||
    { log "check: wrap-check found no CT peaks (empty share ~1 on every wrap): wrong --ct-um $LUM or CT level $K/$LV?"; return 1; }
  mv $O/wrapcheck.tsv.tmp $O/wrapcheck.tsv
  log "check: $(grep -E '^wrap-check: verdicts' $O/logs/check.log | tail -1)"
  # unroll-qa: the label-free sheet-identity gates (wrap-check and the join agreement only measure self-consistency with the ref; DESIGN
  # 2026-11-14): 48 tiles of 128^3 cells, phase confidence / phase-unwrap witness / winding-mod-4 agreement (+ fibre sheet changes with
  # FIBRES=level-0 UFIB); its ok / WARN lines go to progress.log. Warn only; UQSTRICT=1 fails the stage on a gate. ~3-5 min CPU.
  local uqs=(); [ "${UQSTRICT:-0}" = 1 ] && uqs=(--strict); [ -n "${FIBRES:-}" ] && uqs+=(--fibres $FIBRES)
  mkdir -p $O/qa/unroll-qa
  if ! X $O/logs/unrollqa.log $U unroll-qa --fields $F --model $O/joined --ct $R $K/$LV --out $O/qa/unroll-qa --threads $TH --tiles ${UQTILES:-48} "${uqs[@]}"; then
    [ "${UQSTRICT:-0}" = 1 ] && { log "check: unroll-qa failed or a gate failed (UQSTRICT=1; logs/unrollqa.log)"; return 1; }
    log "check: unroll-qa did not finish (logs/unrollqa.log); warn only, continuing"
  fi
  [ "$DRY" = 1 ] || grep -E '^unroll-qa: (ok|WARN) gate' $O/logs/unrollqa.log | while read -r l; do log "check: $l"; done
  # the ref's z drift against the sheets (audit ugeom #5: P4's integers drifted 3 turns over ~5 cm of height and every gate passed):
  # ufsm cf-demod (label-free, CPU; DRIFT=0 off): ALARM above DRIFTWARN turns of along-sheet excursion; DRIFTMAX: fail above it
  if [ "${DRIFT:-1}" = 1 ]; then
    X $O/logs/drift.log $U cf-demod $REF $F --step ${DRIFTSTEP:-8} --band 128 --cz 16 --threads $TH ${DRIFTMAX:+--drift-max $DRIFTMAX}
    local rc=$?; [ "$DRY" = 1 ] && return 0
    local dl; dl=$(grep '^cf-demod: z drift' $O/logs/drift.log | tail -1); log "check: ${dl#cf-demod: }"
    local ex; ex=$(echo "$dl" | sed -nE 's/.*along-sheet excursion ([0-9.]+) turns.*/\1/p')
    [ -n "$ex" ] && awk -v e=$ex -v w=${DRIFTWARN:-0.75} 'BEGIN { exit !(e > w) }' && log "check: ALARM: the ref drifts $ex turns along the sheets over z (> DRIFTWARN ${DRIFTWARN:-0.75}): sheet identity changes with height"
    [ $rc = 3 ] && { log "check: FAILED: z drift above DRIFTMAX $DRIFTMAX"; return 1; }
  fi
  # GT boxes (GTBOXES = a wbench-format tsv in fields-native voxels, GTCODES = "ROOT KEY"; off unless both are set): the joined model and
  # the ref scored on them (CPU): the only check here that sees sheet identity (wrap-check and the join agreement pass P4 at GT ids 0.18)
  if [ -n "${GTBOXES:-}" ] && [ -n "${GTCODES:-}" ] && [ "$DRY" != 1 ]; then
    local gb=$REPO/scripts/spiralqa/ugeom/gtboxes.sh m
    for m in joined ref-cf; do MKCROP=$MKCROP bash $gb $F $O/$m $GTBOXES $GTCODES $O/qa/gtboxes-$m $U > $O/logs/gtboxes-$m.log 2>&1 &&
      log "check: GT boxes, $m: $(tail -1 $O/logs/gtboxes-$m.log)" || log "check: GT boxes, $m: failed (logs/gtboxes-$m.log)"; done
  fi
}
wraps() {
  local zr=; [ -n "$ZR" ] && [ "$ZR" != - ] && zr="--zr $ZR"
  [ "$DRY" = 1 ] || rm -rf $O/wraps
  # the masked-CT gate (geo-outer 2026-11-14; the one no-data alarm): wraps drops roots on weak data cells (the fields' edge on a zeroed
  # CT); a share > --nodata-warn 0.02 means a mask that clips papyrus (the fields want the unmasked / filled CT). NODATAMAX F: exit 3 above
  local nm=; [ -n "${NODATAMAX:-}" ] && nm="--nodata-max $NODATAMAX"
  X $O/logs/wraps.log $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH $zr --min-cm2 ${MINCM2W:-0.5} $nm || return 1
  [ "$DRY" = 1 ] || { grep -q 'wraps: WARNING: no-data' $O/logs/wraps.log && log "wraps: ALARM $(grep -o 'no-data share.*' $O/logs/wraps.log | tail -1)"; return 0; }
}
# trim: both ends of the strip. Wrap ids increase OUTWARD (the strip walks outermost first). From each end, wraps are dropped while
#   (1) they lie beyond the wrap-check table (ids above its highest / below its lowest row: too few radial intervals to judge), or
#   (2) their gap to the next wrap outward is collapsed: wrap-check's rnet_med (the normal gap over the network's own sheet spacing sp) < RNETMIN
#       (0.6): wraps packed closer than the sheets = the ref's over-count pile at the mask edge, or
#   (3) they are near-empty: on-papyrus (surf-stats --ct-only, CT >= 80 within +-15 um) <= PAPMIN (0.1).
# The first kept ids become strip --first / --last (state/strip.args); the inputs per examined wrap go to state/trim.tsv and the dropped
# ids to progress.log. Before 2026-11-14 the outer end used wrap-check's "dup" verdict (empty share >= 0.5), whose precision against the
# 1667 GT is 0.07: it dropped w040-w042 = published turns 29-31; rnet keeps them (rnet .88 / .79 / .66) and drops the pile above (w043 .56,
# w044 .45, w045 .36), and the real outer gaps that shrink with the papyrus (gap 230 -> 94 um, rnet ~1) are no longer read as
# duplicates. The inner core wraps of 1667 (w000-w011, inside the published unwrap's first turn) are on papyrus (0.16-0.56, as the middle
# wraps) at rnet ~1: kept (the published unwrap starts at r 3.9 mm; they are not a fit error). STRIPARGS (env) with its own --last (must
# name a wrap of this run) skips the outer end, a --first there (e.g. Paris 4's tsv --first 78) the inner end; TRIM=0 skips both.
# CLEAN=1 deletes the field planes + slab fits after the snap stage.
onpap() {   # onpap ID..: "id share" per wrap (surf-stats --per-wrap --ct-only on a link dir of those wraps)
  local d=$O/trim.tmp i; rm -rf $d; mkdir -p $d; for i in "$@"; do ln -s $O/wraps/$(printf 'w%03d' $i).sfc $d/; done
  nice -n 10 $U surf-stats $d --fields $F --ct $R $K/$LV --ct-um $LUM --ct-only 1 --per-wrap --threads $TH 2>> $O/logs/trim.log |
    tee -a $O/logs/trim.log | awk '$1 ~ /^w-?[0-9]+$/ && NF >= 8 { print substr($1, 2) + 0, $NF }'
  local rc=${PIPESTATUS[0]}; rm -rf $d; return $rc
}
# trimwalk END ID..: walk the ids from one end (given in walking order), drop while rule 1 / 2 / 3 holds; sets TW_KEPT (none: all dropped)
# and TW_DROP (no subshell: the on-papyrus shares stay in P);
# on-papyrus in batches of 1, 4, then 16 wraps (usually the first wrap past the pile decides). Reads V / RN / LO / HI, appends to state/trim.tsv
trimwalk() {
  local end=$1 k=0 i why nb=1 batch a b dropped=; shift; local ids=("$@")
  while [ $k -lt ${#ids[@]} ]; do
    i=${ids[$k]}; why=
    if [ -z "${RN[$i]:-}" ] && { [ $i -gt $HI ] || [ $i -lt $LO ]; }; then why=unjudged
    elif [ -n "${RN[$i]:-}" ] && awk -v r=${RN[$i]} -v m=${RNETMIN:-0.6} 'BEGIN { exit !(r < m) }'; then why=collapsed
    else
      if [ -z "${P[$i]:-}" ]; then batch=("${ids[@]:$k:$nb}"); nb=$(( nb < 16 ? nb * 4 : 16 ))
        # (a failed surf-stats, or no row for this wrap, fails the trim: a missing share read as 0 dropped the wrap as near-empty, audit
        # mini/shell #2)
        onpap "${batch[@]}" > $O/trim.pap.part || { log "trim: surf-stats failed on w$(printf %03d $i) (+$((${#batch[@]} - 1))), $O/logs/trim.log: refusing"; return 1; }
        while read -r a b; do P[$a]=$b; echo "$a $b" >> $O/trim.pap; done < $O/trim.pap.part; rm -f $O/trim.pap.part
        [ -n "${P[$i]:-}" ] || { log "trim: surf-stats gave no row for w$(printf %03d $i) ($O/logs/trim.log): refusing"; return 1; }; fi
      awk -v p=${P[$i]} -v m=${PAPMIN:-0.1} 'BEGIN { exit !(p <= m) }' && why=empty
    fi
    printf '%s\tw%03d\t%s\t%s\t%s\t%s\t%s\n' $end $i "${V[$i]:--}" "${RN[$i]:--}" "${GP[$i]:--}" "${P[$i]:--}" "${why:-kept}" >> $O/state/trim.tsv
    [ -z "$why" ] && break
    dropped+=" $(printf 'w%03d' $i)($why)"; k=$((k + 1))
  done
  TW_KEPT=none; [ $k -lt ${#ids[@]} ] && TW_KEPT=${ids[$k]}; TW_DROP=$dropped
}
trim() {
  [ "$DRY" = 1 ] && { echo "    strip --first / --last = the first kept id from each end: beyond the wrap-check table, rnet_med < ${RNETMIN:-0.6} (collapsed gaps), on-papyrus <= ${PAPMIN:-0.1} (surf-stats --ct-only) dropped"; return 0; }
  local args=$STRIPARGS ids w r first last o TW_KEPT TW_DROP; : > $O/state/strip.args; : > $O/trim.pap
  printf 'end\twrap\tverdict\trnet_med\tgap_um_med\ton_papyrus\tdecision\n' > $O/state/trim.tsv
  if [[ " $STRIPARGS " == *" --last "* ]]; then   # an explicit STRIPARGS --last (env): must name a wrap of THIS run
    last=$(echo " $STRIPARGS " | sed -nE 's/.* --last (-?[0-9]+) .*/\1/p')
    [ -e $O/wraps/$(printf 'w%03d' $((10#$last))).sfc ] || { log "trim: STRIPARGS --last $last names no wrap of this run ($(ls $O/wraps | head -1)..$(ls $O/wraps | tail -1)): a --last from another run's gauge? refusing"; return 1; }
  fi
  if [ "${TRIM:-1}" != 1 ]; then log "trim: skipped (TRIM=${TRIM:-1}, STRIPARGS '$STRIPARGS')"; echo "$args" > $O/state/strip.args; return 0; fi
  ids=($(ls $O/wraps | sed -n 's/^w\(-\?[0-9]*\)\.sfc$/\1/p' | awk '{print $1 + 0}' | sort -n)); [ ${#ids[@]} -gt 0 ] || { log "trim: no wraps"; return 1; }
  declare -A V RN GP P; LO=; HI=
  while IFS=$'\t' read -r w _ _ _ _ _ r g _ _ _ _ _ v; do [[ $w =~ ^w-?[0-9]+$ ]] || continue; w=${w#w}; [[ $w == -* ]] && w=$(( -10#${w#-} )) || w=$((10#$w)); V[$w]=$v; RN[$w]=$r; GP[$w]=$g
    { [ -z "$LO" ] || [ $w -lt $LO ]; } && LO=$w; { [ -z "$HI" ] || [ $w -gt $HI ]; } && HI=$w; done < $O/wrapcheck.tsv
  [ -n "$HI" ] || { log "trim: wrapcheck.tsv has no wrap rows: refusing (TRIM=0 to skip)"; return 1; }
  if [[ " $STRIPARGS " == *" --last "* ]]; then log "trim: outer end kept from STRIPARGS (--last $last)"
  else
    trimwalk outer $(printf '%s\n' "${ids[@]}" | sort -rn) || return 1; last=$TW_KEPT; o=$TW_DROP
    [ "$last" = none ] && { log "trim: every wrap would be dropped from the outer end: refusing (TRIM=0 to skip)"; return 1; }
    args="$args --last $last"; log "trim: strip --last $last (verdict ${V[$last]:--}, rnet ${RN[$last]:--}, on-papyrus ${P[$last]:--}); outer dropped $(echo $o | wc -w):${o:+ $o}"
  fi
  if [[ " $STRIPARGS " == *" --first "* ]]; then log "trim: inner end kept from STRIPARGS ($STRIPARGS)"
  else
    trimwalk inner "${ids[@]}" || return 1; first=$TW_KEPT; o=$TW_DROP
    [ "$first" = none ] || [ $first -gt $last ] && { log "trim: the inner end passes the outer one ($first > $last): refusing (TRIM=0 to skip)"; return 1; }
    log "trim: inner end: first kept w$(printf %03d $first) (rnet ${RN[$first]:--}, on-papyrus ${P[$first]:--}); inner dropped $(echo $o | wc -w):${o:+ $o}"
    # (ufsm strip reads ids >= 0 only, src/strip.c: the kept negative ids are named here, not silently lost)
    [ $first -lt 0 ] && { log "trim: NOTE: kept wraps w$(printf %03d $first)..w-01 are not in the strip (strip takes ids >= 0)"; first=0; }
    args="$args --first $first"
  fi
  echo "$args" | sed 's/^ *//' > $O/state/strip.args
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
    # only the wraps the strip keeps (trim's --first / --last in state/strip.args; wrap-snap still reads the ids just outside as the
    # neighbours that bound them, so the kept wraps snap byte-identically): Paris 4 dropped 41 outer wraps, 19% of the vertices
    local sel=; [ -e $O/state/strip.args ] && sel=$(grep -oE -- '--(first|last) +-?[0-9]+' $O/state/strip.args | tr '\n' ' ')
    # SNAPGPU (default: the first GPU of FITGPUS that is not INKGPU, free after the slabs; empty: none): the CT / fields chunks decoded
    # there (UFSM_GPU_DECODE, byte-identical to the CPU's; ~45% of wrap-snap's CPU time was the volcomp decode)
    local sg=${SNAPGPU-$(for g in $FITGPUS; do [ "$g" != "$INKGPU" ] && { echo $g; break; }; done)}
    X $O/logs/snap.log env ${sg:+UFSM_GPU_DECODE=1 UFSM_GPU_DECODE_DEV=$sg} $U wrap-snap $O/wraps $O/wraps_snap --target $t --axis $O/axis.json --axis-um $UM --ct $R $K/$LV --ct-um $LUM \
      --fields $F --threads $TH $sel ${MASKMAX:+--masked-max $MASKMAX} ${SNAPX:-} || return 1
    [ "$DRY" = 1 ] || log "snap: $(grep '^wrap-snap: solve' $O/logs/snap.log | tail -1)"
    # (the masked-CT / no-data alarm in progress.log is geoouter-20261114's wraps --fields line; wrap-snap keeps its WARNING print, MASKMAX
    # passes --masked-max: fail above that share of the wrap vertices on masked CT, off by default)
  else log "snap: off (SNAP=0)"; fi
  [ "$DRY" = 1 ] || { [ "${CLEAN:-1}" = 1 ] && { find $F -mindepth 1 -maxdepth 1 ! -name fields.json -exec rm -rf {} +; [ $F = $O/fields/$FNAME ] || find $O/fields/$FNAME -mindepth 1 -maxdepth 1 ! -name fields.json -exec rm -rf {} +; rm -rf $O/slabs/s???/fit; }; }
  return 0
}
strip() { local a=$STRIPARGS; [ -e $O/state/strip.args ] && a=$(cat $O/state/strip.args)
          [ "$DRY" = 1 ] || rm -rf $O/strip/strip; X $O/logs/strip.log $U strip $(wrapsdir) $O/strip/strip --threads $TH $a; }
stripflat() { X $O/logs/stripflat.log $U strip-flatten $O/strip/strip --decim ${DECIM:-4} --threads $TH --flip-rows 1 --cg 60 --tol 3e-4 ${FLATX---gpu 1}; }
# distort: the flattening's distortion gate (src/stripdistort.c; audit 2026-11-14: measured in QA only, never checked; 1299's flatten energy
# 5904 vs P4 5.4 went unnoticed). strip-distort -> strip/distort (per-stripe <s>_dist.jpg / <s>_area.jpg overlays, tiles / wraps / stripes
# .tsv; 1667 9 s, P4 172 s, CPU); the totals and strip-flatten's own summary go to progress.log. FAILS the stage (the strip images wait)
# when dist p95 > DISTP95 (0.35; 1667 .157, P4 .187) or the folded share > DISTFOLD (0.005; 1667 .0018, P4 .0006), or strip-flatten
# skipped a chunk ("no mesh": a hole later filled as if real); DISTGATE=0: measured and logged only
distort() {
  local d=$O/strip/distort t
  [ "$DRY" = 1 ] && { echo "    $U strip-distort $O/strip/strip/flat $d (gate: dist p95 <= ${DISTP95:-0.35}, folded <= ${DISTFOLD:-0.005}, no skipped chunk; DISTGATE=${DISTGATE:-1})"; return 0; }
  rm -rf $d; X $O/logs/distort.log $U strip-distort $O/strip/strip/flat $d --threads $TH || return 1
  t=$(grep -E '^strip-distort: total' $O/logs/distort.log | tail -1); [ -n "$t" ] || { log "distort: no totals (logs/distort.log)"; return 1; }
  log "distort: ${t#strip-distort: }"
  log "distort: strip-flatten: $(grep -E '^strip-flatten: flattened' $O/logs/stripflat.log | tail -1 | sed 's/^strip-flatten: //')"
  local p95 fold nom bad=; p95=$(echo "$t" | sed -nE 's/.* p95 ([0-9.]+) .*/\1/p'); fold=$(echo "$t" | sed -nE 's/.*folded ([0-9.]+).*/\1/p')
  nom=$(grep -c 'no mesh' $O/logs/stripflat.log)
  awk -v v=$p95 -v m=${DISTP95:-0.35} 'BEGIN { exit !(v > m) }' && bad+=" dist p95 $p95 > ${DISTP95:-0.35};"
  awk -v v=$fold -v m=${DISTFOLD:-0.005} 'BEGIN { exit !(v > m) }' && bad+=" folded $fold > ${DISTFOLD:-0.005};"
  [ "$nom" -gt 0 ] && bad+=" strip-flatten skipped $nom chunk(s) (no mesh);"
  [ -z "$bad" ] && return 0
  if [ "${DISTGATE:-1}" = 1 ]; then log "distort: GATE FAILED:$bad see strip/distort/*_dist.jpg (DISTGATE=0 to go on)"; return 1; fi
  log "distort: over the gate (DISTGATE=0, going on):$bad"; return 0
}
# stripimg: the filled stripe renders + two mosaics (the pods' strip.sh recipe): a contrast stretch from the 3 middle stripes (rendered
# with fill into strip/render_fill, reused), then scripts/strip/strip_refill.sh: every stripe rendered with FILLX, strip-mosaic img_fill
# (grey <s>_ct.jpg + <s>_ink.jpg per stripe) and img_pap (papyrus look, --papyrus-dir, same reads), renders removed after the mosaic. UFSM_RENDER_GPU_MEM 4 (GB):
# the render's device chunk pool stays small enough for a slab fit sharing the GPU (pods, two scrolls); RGPU: CUDA device of the renders
# FILLCT (80 by default since 2026-11-14, audit-urender #2; 1 = the old behaviour: only masked-zero CT is void): 80 recentres CT + ink on the nearest papyrus
# within --fill voxels wherever the surface sample is below the mosaic's papyrus level (audit-urender 2026-11-14: 45-54 % of 1667's
# on-surface render px have CT < 80 at the centre surface)
FILLX=${FILLX:---fill 4 --fill-ct ${FILLCT:-80} --fill-grid 50000 --fill-inpaint 4 --fill-pm 0}
# inkgate: the latent ink head's c12 / c13 (p_recto / p_verso) beside the ink, for strip_refill.sh's papyrus gate (the latent-ink policy:
# blended + papyrus-gated); none (ift3, or a run without them): ungated. INKGATE=0 off
inkgate() { [ "${INKGATE:-1}" = 0 ] && return 0; [ -e $O/ink/c12.zarr/zarr.json ] && [ -e $O/ink/c13.zarr/zarr.json ] && echo "$O/ink/c12.zarr $O/ink/c13.zarr"; return 0; }
stripimg() {
  local ST=$O/strip RF=$O/strip/render_fill S N k s
  local E="UFSM_ZCACHE_GB=$ZCACHE_GB UFSM_ZCACHE_ADMIT=${ZCACHE_ADMIT:-0.6} UFSM_JPEG_THREADS=$JPEGT UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4} UFSM_RENDER_MEM=${UFSM_RENDER_MEM:-16} ${RGPU:+CUDA_VISIBLE_DEVICES=$RGPU}"
  if [ "$DRY" = 1 ]; then echo "    $E: stretch from the 3 middle stripes (render $FILLX + strip-mosaic --stretch-only), then U=$U CTR=$R CTK=$K/$LV INK=$INK INKK=$LUM PAPMOS=$ST/img_pap bash scripts/strip/strip_refill.sh $ST"; return 0; fi
  mkdir -p $RF $ST/fillmap $ST/logs; S=($(ls $ST/strip/flat/stripes)); N=${#S[@]}; [ $N -gt 0 ] || return 1
  if [ ! -s $ST/stretch.txt ]; then
    # the 3 middle stripes' renders at once (SJOBS, default by memory: a Paris 4 stripe render peaks at ~21 GB RSS and 4 GB of device
    # pool; one after another they were 3 x 28 s of 84 s, on 1667 26 + 15 + 10 s; each render is independent: the same images)
    local sj=${SJOBS:-$(a=$(availgb); [ $a -ge 72 ] && echo 3 || { [ $a -ge 48 ] && echo 2 || echo 1; })} pids=() rc=0 p
    srend() { local s=$1; mkdir -p $RF/in_$s; ln -sfn $ST/strip/flat/stripes/$s $RF/in_$s/$s
      env $E /usr/bin/time -v nice -n 10 $U render $RF/in_$s $RF --volume $INK $LUM --ct $R $K/$LV --depth 1 --scale 1 --threads ${RENDER_NT:-$TH} $FILLX > $ST/logs/refill_$s.log 2>&1 || return 1
      [ -e $RF/${s}_fill.tif ] && mv $RF/${s}_fill.tif $ST/fillmap/; rm -r $RF/in_$s; touch $RF/$s.done; }
    for k in $((N / 2 - 1)) $((N / 2)) $((N / 2 + 1)); do [ $k -ge 0 ] && [ $k -lt $N ] || continue; s=${S[$k]}; [ -e $RF/$s.done ] && continue
      while [ ${#pids[@]} -ge $sj ]; do wait ${pids[0]} || rc=1; pids=("${pids[@]:1}"); done
      srend $s & pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || rc=1; done
    [ $rc = 0 ] || return 1
    nice -n 10 $U strip-mosaic $ST/strip/flat $RF $ST/img_fill --stretch-only 1 --threads 3 2> $ST/logs/stretch.log | awk '/^stretch/ {print $2}' > $ST/stretch.txt
    [ -s $ST/stretch.txt ] || { log "stripimg: no stretch (logs in $ST/logs)"; return 1; }; log "stripimg: stretch $(cat $ST/stretch.txt)"
  fi
  # RJOBS: 2 stripe renders at once when >= 64 GB is available (0490B: 394 -> 355 s, 6.7 -> 18.4 GB RSS, same images), else 1
  local rj=${RJOBS:-$([ $(availgb) -ge 64 ] && echo 2 || echo 1)}
  env $E U=$U CTR=$R CTK=$K/$LV INK=$INK INKK=$LUM RJOBS=$rj RENDER_NT=${RENDER_NT:-$TH} MOS_NT=${MOS_NT:-4} MAXR=${MAXR:-6} MEMGB=${RMEMGB:-0} \
    PAPMOS=$ST/img_pap PGAIN=${PGAIN:-1} PAPPX=${PAPPX:-8000} FILLX="$FILLX" INKGATE="$(inkgate)" /usr/bin/time -v bash $REFILL $ST >> $O/logs/stripimg.log 2>&1 &&
    [ -e $ST/img_fill/overview.jpg ] && log "stripimg: $(grep '^strip-mosaic: .* overview' $ST/logs/mosaic_fill.log | head -1)"
}
# --- per-wrap renders (p4full3.sh) and faces ---
pw_one() {   # pw_one WRAP GPU (a background subshell: the cache env stays in it); the wraps the strip uses (wraps_snap with SNAP=1, audit 11-14 #12)
  zcon; local w=$1 gpu=$2 L=$O/logs/pw/$1 nv dc d=$O/r/$1 WD=$(wrapsdir)
  nv=$(stat -c %s $WD/$w.sfc); dc=$(awk -v n=$nv -v m=1500000 'BEGIN { d = 1; while (n / (d * d) > m) d++; print d }')
  if [ ! -e $O/flat/$w/x.tif ]; then
    nice -n 10 $U tifxyz-scale $WD $O/orig 1 --wraps $w > $L.scale.log 2>&1 || return 1
    rm -rf $O/flat/$w
    CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $U flatten $O/orig $O/flat --wraps $w --decim $dc --threads ${FTH:-4} --crop none --gpu 1 > $L.flat.log 2>&1 || return 1
  fi
  nice -n 10 $U tifxyz-scale $O/flat $O/up 1 --wraps $w --flip-rows 1 > /dev/null 2>&1 || return 1
  mkdir -p $d/in; ln -sfn $O/up/$w $d/in/$w
  UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 UFSM_RENDER_GPU_MEM=${UFSM_RENDER_GPU_MEM:-4} CUDA_VISIBLE_DEVICES=$gpu nice -n 10 $U render $d/in $d --volume $INK $LUM --ct $R $K/$LV --depth 1 --scale 1 --threads ${RTH:-8} > $d/render.log 2>&1 || return 1
  nice -n 10 $U ink-compare $O/img/${w}_full_side.jpg --ct $d/${w}_ct.tif --ink "$INKNAME=$d/$w.tif" --px-um $LUM --stretch --stretch-ct --max-px 65000 --threads ${RTH:-8} \
    --then $O/prev/${w}_side.jpg --max-px 8000 > $d/stats.txt 2>&1 || return 1   # (its status is the job's: audit mini/shell #4)
  [ -e $O/img/${w}_full_side.jpg ] && rm -f $d/$w.tif $d/${w}_ct.tif
  echo "$(date '+%F %T') $w $(grep -E '^[^ ]+: input' $L.flat.log | head -1 | grep -oE '= [0-9.]+ cm2')" >> $O/logs/perwrap.list
}
perwrap() {
  [ "$DRY" = 1 ] && { echo "    per wrap (largest first, SPAR ${SPAR:-4}, GPUs of FITGPUS in turn): tifxyz-scale -> flatten --gpu 1 -> tifxyz-scale --flip-rows 1 -> render (ink + CT) -> ink-compare --then"; return 0; }
  mkdir -p $O/orig $O/flat $O/up $O/r $O/img $O/prev $O/logs/pw
  local w k=0 nf=0 p pg=($FITGPUS) WD=$(wrapsdir); local -A pid
  for w in $(for f in $WD/w*.sfc; do w=$(basename $f .sfc); [ -e $O/img/${w}_full_side.jpg ] || echo "$(stat -c %s $f) $w"; done | sort -rn | awk '{print $2}'); do
    stopped && break
    while [ $(jobs -r | wc -l) -ge ${SPAR:-4} ]; do sleep 5; done; memok
    pw_one $w ${pg[$((k % ${#pg[@]}))]} & pid[$w]=$!; k=$((k + 1))
  done
  for w in "${!pid[@]}"; do wait ${pid[$w]} || { nf=$((nf + 1)); echo "perwrap: $w FAILED (logs $O/logs/pw/$w.*, $O/r/$w/render.log)" >> $O/logs/perwrap.log; }; done   # (each job's rc, audit 11-14 #12)
  echo "perwrap: $(ls $O/img/*_full_side.jpg 2>/dev/null | wc -l) wraps rendered from $WD, $nf of $k failed" >> $O/logs/perwrap.log
  [ $nf = 0 ] && ! stopped   # (a STOP leaves wraps unrendered: no perwrap.done, the rerun renders them)
}
faces() {
  [ "$DRY" = 1 ] && { echo "    ufsm faces per flattened wrap ($O/flat) -> $O/faces (faces_p4e.sh recipe)"; return 0; }
  mkdir -p $O/faces; local w nf=0; zcon
  for w in $(ls $O/flat); do [ -e $O/flat/$w/x.tif ] || continue; [ -e $O/faces/out/${w}_recto ] && continue; stopped && return 1; memok
    X $O/logs/faces.log $U faces $O/flat $O/faces/out --wraps $w --ct $R $K/$LV --ct-um $LUM --threads ${RTH:-8} || { nf=$((nf + 1)); echo "faces $w failed" >> $O/logs/faces.log; }
  done
  # (any failed wrap fails the stage, so faces.done is not written and a rerun retries them; audit mini/shell #4)
  [ $nf = 0 ] || { log "faces: $nf wraps failed ($O/logs/faces.log)"; return 1; }
}
upload() {
  local cmd="ssh desk ~/reg-upload/upcorpus.sh $S   # + lftp mirror -R of wraps / strip img (scripts/corpus/upcorpus.sh)"
  case " $PRIZE " in *" $S "*) log "upload: $S is prize-eligible: NOT uploaded (ask first)"; return 0;; esac
  if [ "${UPLOAD:-0}" != 1 ]; then log "upload: not run (UPLOAD=1 to run): $cmd"; return 0; fi
  X $O/logs/upload.log bash $REPO/scripts/corpus/upcorpus.sh $S
}

# --- the schedule ---
[ "$DRY" = 1 ] && echo "unroll $S: O=$O U=$U ($R $K level $LV, $LUM um; box ${BOX:--}, zr ${ZR:--}, strip args '${STRIPARGS}'; CK $CK: $FKIND fields $F)"
mstart; stage prep prep || exit 1; upto prep
stage fields fields || exit 1; upto fields
# the ink on GPU $INKGPU. INKAT=slabs (default): its GroupNorm calibration beside the ref, the main pass from the end of the slabs, beside
# the CPU stages join .. stripflat (P4 11-12: ink 892 s; with the fast ref its GPU-1 slab lane would wait for it); INKAT=fields: the
# whole ink from the end of the fields, the GPU-1 slab lane waiting for it (before 11-12). The per-wrap renders beside the strip
# INKWAIT=0 (with INKAT=fields): the lanes on INKGPU do not wait for the ink, they share the card (one-GPU desk runs: the ink, ~13 min on
# 0332, would otherwise hold the whole slab stage back)
# (a rerun of the same O clears the last run's .failed markers first: waitfor / the GPU lane stopped at once on a stale one, audit mini/shell #3)
[ "$DRY" = 1 ] || rm -f $O/state/ink.failed $O/state/inkcal.failed
# one-GPU runs (FITGPUS = INKGPU, the desk): INKAT=fields INKWAIT=0 by default (0332 2026-11-17, card 1 only: 1832 -> 1422 s end to end;
# from the slabs the ink, 770 s, was the critical path and stripimg / perwrap waited 8.5 min for it; beside rectosp / ref / slabs those
# slow down 128 -> 221, 54 -> 92, 149 -> 190 s but the ink is done before the snap)
[ -z "${INKAT:-}" ] && [ "$(echo $FITGPUS)" = "$INKGPU" ] && { INKAT=fields; INKWAIT=${INKWAIT:-0}; }
INKAT=${INKAT:-slabs}; inkbg() { rm -f $O/state/ink.failed; ( stage ink ink || touch $O/state/ink.failed ) & INKP=$!; [ "$DRY" = 1 ] && wait $INKP; return 0; }
if [ "${NOINK:-0}" != 1 ]; then
  if [ "$INKAT" = fields ]; then inkbg; else rm -f $O/state/inkcal.failed; ( stage inkcal inkcal || touch $O/state/inkcal.failed ) & [ "$DRY" = 1 ] && wait; fi
fi
inkmain() { [ "${NOINK:-0}" != 1 ] && [ "$INKAT" != fields ] && ! done_ ink && inkbg; return 0; }
{ [ "${RECTOSP:-1}" = 0 ] || stage rectosp rectosp; } || { log "unroll: rectosp failed"; wait; exit 1; }
[ "${RECTOSP:-1}" != 0 ] && { [ -e $O/fields/rsp/fields.json ] || { [ "$DRY" = 1 ] && [ -n "$(m7src)" ]; }; } && F=$O/fields/rsp   # (the ref and every later stage read the blended fields)
{ [ "${CTCAL:-0}" != 1 ] || stage ctcal ctcal; } && stage ref ref && upto ref && stage slabs slabs && inkmain && upto slabs && stage join join && upto join && stage check check && upto check &&
  stage wraps wraps && upto wraps && stage trim trim && stage snap snap && upto snap ||
  { log "unroll: stopped before the strip"; wait; exit 1; }
if [ "${PERWRAP:-0}" = 1 ]; then ( waitfor ink && stage perwrap perwrap && { [ "${FACES:-0}" = 1 ] && stage faces faces; true; } ) & PWP=$!; [ "$DRY" = 1 ] && wait $PWP; fi
stage strip strip && upto strip && stage stripflat stripflat && stage distort distort && waitfor ink && stage stripimg stripimg || log "unroll: the strip stopped"
wait
[ "$DRY" = 1 ] || [ -n "${CTCACHE:-}" ] || [ "${KEEPCT:-0}" = 1 ] || { case "$O/ctcache" in /*/ctcache) rm -rf -- "$O/ctcache";; esac; }
stage upload upload
[ "$DRY" = 1 ] || log "unroll $S finished: $(for f in $O/state/*.done; do [ -e $f ] && printf '%s %s; ' $(basename $f .done) "$(cut -d, -f1 $f)"; done)"

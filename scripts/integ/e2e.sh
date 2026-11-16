#!/usr/bin/env bash
# scripts/integ/e2e.sh UFSM_BINARY: one repeatable end-to-end integration run of a release (mcc) build, on one GPU, in minutes per step.
#
#   1  train      the next production run's argv (scripts/run/scratch2/try7g0.launch.env + scratch2.launch + scratch2.train.sh, mirrored below;
#                 the argv-sync row diffs it against the last production trainer argv in $R/scratch2.resume.log) at P $P for $STEPS1 steps
#                 (--stop-at, the run's schedule is $STEPS long) on Paris 4 only, windows inside TRAINBOX (level-0 voxels; it clears every box of
#                 the run's holdouts-20261114.json by >= 128 voxels; HOLD, the eval box, is 2 k voxels below it). Checks: rc 0, the step reached,
#                 no nan / inf on a step / val line, the mean loss of the last 3 logged steps < LOSSDROP x that of the first 3, last.ckpt at that step.
#   2  resume     the same argv + --resume last.ckpt (as scratch2.train.sh relaunches): rc 0, its first step > STEPS1, reaches STEPS, finite, its mean
#                 loss <= RESUMEFAC x run 1's last-3 mean (an optimizer state lost on resume shows as a jump), last.ckpt at STEPS.
#   3  e2e-pred   the 200-step net (--ema 0: the EMA lags ~1000 steps): fit fields + every head (predict --heads all) on HOLD, and the wbench box fit
#                 scored against the GT codes. A smoke test of predict / fit on a fresh checkpoint of this build: its ids are printed, not gated (a
#                 40-update net is not a winding model; count-field / wraps on it would fail for the wrong reason).
#   4  ref-box    the pinned reference snapshot REFCK (try6-2357, scratch2 try6 at step 2357, the newest latent production net with known-good GT
#                 numbers) on HOLD with the exact wbench recipe of its reference row (WB/res/REFTAG.tsv, REFTAG try6-gnlive: cube window 640 / 64,
#                 --ema 1, --gn-live 1, fit --dir 1 --uw-repair 8 --codes): ids0 / p05x / p1x within IDSTOL / PTOL of that row. HOLD is held out of
#                 every run (wbench boxes.tsv) and labelled (paris4-q9p6 codes).
#   5..11 chain   REFCK on CHAINBOX (HOLD widened in y / x so the umbilicus sits inside: the count-field reference counts from the axis):
#                 fields + heads -> count-field ref (spiral dir from scripts/unroll/spiral_dir.tsv) -> 2 z slab fits snapped to the ref (unroll.sh
#                 lane recipe) -> fit-join --ref, the joined model scored against the GT codes, wrap-check -> wraps -> flatten (largest wrap) ->
#                 render (ink head + CT) + ink-compare. Gated against the baseline file BASE (written by a run with BLESS=1, all rows PASS; a
#                 STEPS_ONLY run blesses only its own keys); with no baseline only MINWRAPS (and MINJOINED when set) apply. Each step runs when
#                 its input exists, so a metric miss upstream never skips the rest of the chain.
#   Prints step | result | numbers | log, writes OUT/results.tsv + OUT/metrics.tsv; exit 0 only when no row is FAIL / SKIP (WARN passes).
#
# Env: GPU (1: the CUDA device; desk card 0 runs scratch2), OUT (/vesuvius/ufsm/integ/e2e/<date>-<binary sha>; must be under /vesuvius),
#   P (256), STEPS1 (200), STEPS (240), LOSSDROP (0.95), RESUMEFAC (1.3), WORKERS (6), SLOTS (4), LAUNCHENV (try7g0.launch.env of this tree),
#   REFCK (/vesuvius/ufsm/audit-20261114/eval-scratch/try6-2357.ckpt), REFTAG (try6-gnlive), IDSTOL (0.05), PTOL (0.03), CHAINCK (ref | e2e),
#   BASE (/vesuvius/ufsm/integ/e2e/baseline-<REFTAG>.tsv), BLESS (0; 1 writes BASE from this run when every row passed), BTOL (0.05),
#   MINJOINED (unset: no absolute floor; 0.40 failed a good join, 11-16), MINWRAPS (3), TH (12 CPU threads), MEMGB (36: ~/bin/memcap -m, the dev slice),
#   LOCK (comma list of flock files held for the whole run; default this GPU's /vesuvius/ufsm/opt + /tmp/ufsm-suite-locks locks), LOCKWAIT (s, 7200),
#   BUSYOK (0; 1 runs although another process holds > 200 MiB of the card), FORCE (1: reuse an existing OUT, its step dirs removed),
#   CLEAN (1: the checkpoint ring / snapshots removed after the checks; last.ckpt kept), STEPS_ONLY ("" = all; e.g. "ref-box chain").
# Never copies data: every CT / label / checkpoint path is read in place. Runs with bash (the desk shell is zsh).
set -u -o pipefail
U=$(readlink -f "${1:?usage: e2e.sh UFSM_BINARY}"); [ -x "$U" ] || { echo "e2e: $U is not an executable" >&2; exit 2; }
H=$(cd "$(dirname "$0")/../.." && pwd)
GPU=${GPU:-1}; P=${P:-256}; STEPS1=${STEPS1:-200}; STEPS=${STEPS:-240}; TH=${TH:-12}
BSHA=$(sha1sum "$U" | cut -c1-8)
OUT=${OUT:-/vesuvius/ufsm/integ/e2e/$(date +%Y%m%d-%H%M%S)-$BSHA}
case "$OUT" in /vesuvius/?*/?*) ;; *) echo "e2e: OUT must be a directory at least two levels under /vesuvius (got '$OUT')" >&2; exit 2;; esac
OUT=${OUT%/}; O=$OUT; LG=$O/logs
WB=${WB:-/vesuvius/ufsm/wbench}
REFCK=${REFCK:-/vesuvius/ufsm/audit-20261114/eval-scratch/try6-2357.ckpt}; REFTAG=${REFTAG:-try6-gnlive}
BASE=${BASE:-/vesuvius/ufsm/integ/e2e/baseline-$REFTAG.tsv}
LAUNCHENV=${LAUNCHENV:-$H/scripts/run/scratch2/try7g0.launch.env}
PRODLOG=${PRODLOG:-/vesuvius/ufsm/v2/lat/runs/scratch2.resume.log}

# ---- data (all read in place)
CTR=/vesuvius/ufsm/v2/ct; CTK=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr; CTUM=2.4; LV=2; LUM=9.6
AX=/vesuvius/ufsm/teacher/axis-0.json                       # (wbench site-desk.env AXP4 fallback: the Paris 4 umbilicus, level-0 voxels)
QR=/vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr; QK=9.6          # GT winding codes (wbench scroll paris4)
HOLDBOX=12288,4096,3584,512,576,768                           # wbench boxes.tsv HOLD, level-2 voxels ("held out by all runs")
CHAINBOX=${CHAINBOX:-12288,3584,3584,512,1280,1280}           # HOLD + the axis (level 2: y ~4056..4334, x ~4190..4240 over these z); 512-aligned (same tile grid)
TRAINBOX=${TRAINBOX:-53248,13392,12996,2688,8192,8192}        # level 0: training windows only here (sources "boxes"); labelled (codesA48 ~4 % sheet voxels)
E2E_SOURCES=${E2E_SOURCES:-PHercParis4-june PHercParis4-u2ink PHercParis4-fib}   # phase / normal / density / wind4 / rv, ink (onfly segments), fibre
SCROLL=PHercParis4

mkdir -p "$O" || exit 2
saferm() {   # saferm PATH...: rm -rf only strictly inside OUT (zsh-proof: called from bash, each path quoted)
  local p; for p in "$@"; do
    case "$p" in "$O"/?*) case "$p" in *..*) echo "e2e: refusing to remove '$p' (..)" >&2; exit 3;; esac; rm -rf -- "$p";;
      *) echo "e2e: refusing to remove '$p' (outside $O)" >&2; exit 3;; esac; done; }
if [ -n "$(ls -A "$O" 2>/dev/null)" ]; then
  [ "${FORCE:-0}" = 1 ] || { echo "e2e: $O is not empty (FORCE=1 reuses it, its step outputs removed)" >&2; exit 2; }
  for d in train hold-e2e hold-ref chain logs results.tsv metrics.tsv sources-e2e.json; do [ -e "$O/$d" ] && saferm "$O/$d"; done
fi
mkdir -p "$LG"
RES=$O/results.tsv; MET=$O/metrics.tsv; : > "$RES"; : > "$MET"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LG/e2e.log" >&2; }
row() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RES"; log "[$2] $1: $3 ($4)"; }   # row STEP RESULT NUMBERS LOG
met() { printf '%s\t%s\n' "$1" "$2" >> "$MET"; }
num() { [[ "${1:-}" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]] && echo "$1" || echo na; }
want() { [ -z "${STEPS_ONLY:-}" ] || [[ " $STEPS_ONLY " == *" $1 "* ]]; }
ok_of() { awk -F'\t' -v s="$1" '$1 == s { r = $2 } END { exit !(r == "PASS" || r == "WARN") }' "$RES"; }
awkc() { awk "BEGIN { exit !($*) }"; }   # awkc 'EXPR': a float comparison

log "e2e: binary $U ($BSHA), GPU $GPU, out $O"
# ---- preflight: data, disk, GPU locks, the card free
for f in "$CTR/$CTK/2/zarr.json" "$AX" "$QR/$QK/zarr.json" "$REFCK" "$LAUNCHENV" "$H/scripts/wbench/mkcrop.sh" "$H/scripts/unroll/spiral_dir.tsv"; do
  [ -e "$f" ] || { echo "e2e: missing $f" >&2; exit 2; }; done
fgb=$(df -BG --output=avail "$O" | tail -1 | tr -dc 0-9); [ "${fgb:-0}" -ge ${MINFREEGB:-30} ] || { echo "e2e: only ${fgb} GB free on $O (MINFREEGB ${MINFREEGB:-30}; diskclean)" >&2; exit 2; }
LOCK=${LOCK-/vesuvius/ufsm/opt/gpu$GPU.lock,/tmp/ufsm-suite-locks/gpu$GPU.0,/tmp/ufsm-suite-locks/gpu$GPU.1,/tmp/ufsm-suite-locks/gpu$GPU.excl}
if [ -n "$LOCK" ]; then mkdir -p /tmp/ufsm-suite-locks 2>/dev/null; fd=20
  for lk in ${LOCK//,/ }; do eval "exec $fd>>\"$lk\"" || exit 2; log "waiting for $lk"; flock -w ${LOCKWAIT:-7200} $fd || { echo "e2e: no lock $lk within ${LOCKWAIT:-7200} s" >&2; exit 2; }; fd=$((fd + 1)); done
  log "holding $LOCK"; fi
if [ "${BUSYOK:-0}" != 1 ] && command -v nvidia-smi > /dev/null; then
  busy=$(nvidia-smi --id=$GPU --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$2 > 200 {print $1 "(" $2 " MiB)"}' | tr '\n' ' ')
  [ -z "$busy" ] || { echo "e2e: GPU $GPU is in use by $busy (BUSYOK=1 to share it)" >&2; exit 2; }
fi
MC=(); [ -x "$HOME/bin/memcap" ] && MC=("$HOME/bin/memcap" -m ${MEMGB:-36}G)   # (the dev slice; trainers of production run in -t)
X() {   # X LOG TIMEOUT_S CMD...: CMD on GPU $GPU (CUDA_VISIBLE_DEVICES; the commands get --gpu(s) 0), memory-capped, logged
  local lg=$1 t=$2; shift 2; echo "+ $*" > "$lg"
  timeout -k 60 "$t" env CUDA_VISIBLE_DEVICES=$GPU MALLOC_ARENA_MAX=4 nice -n 5 "${MC[@]}" "$@" >> "$lg" 2>&1; }
ckstep() { head -c 200000 "$1" 2>/dev/null | head -1 | grep -ao '"step":[0-9]*' | head -1 | cut -d: -f2; }
gnflag() { head -c 200000 "$1" | head -1 | grep -aq '"gn_ema":1' && echo "--gn-live 1"; }   # (gn_ema nets train on window statistics: --gn-ema-warm 1e9)

# ---- the production argv (mirror of scratch2.launch ARCH / EXTRA and scratch2.train.sh AUG / HEADS / ARGS; keep in sync, argv-sync checks it)
declare -A LE; while IFS= read -r l; do case "$l" in ''|'#'*) continue;; esac; [[ "$l" == *=* ]] && LE[${l%%=*}]=${l#*=}; done < "$LAUNCHENV"
le() { [ -n "${LE[$1]+x}" ] && echo "${LE[$1]}" || echo "$2"; }
read -ra OPTX <<< "$(le OPTX '')"; read -ra XFLAGS <<< "$(le XFLAGS '')"
SRCJ=$(le SRCJ /vesuvius/ufsm/univ/scratch1/sources-scratch1.json)
TRAIN_ENV=(UFSM_HOSTMEM=1 UFSM_HEAD_SLAB_Z=$(le HEADSLABZ 8))
TARGS=(--gpus 0 --P $P --B 1 --widths ${WIDTHS:-256,320,384,448,288,288} --attn-blocks 4 --attn-pe scroll --gn-ema 1 --scan-norm 1 --gfilm 1 --fibre-gpu 0 --ink-gpu 0
  --task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1
  --steps $STEPS --sched wsd --cooldown 0.2 --opt muon --fp4 2 --gpu-aug 3 --mem auto --val-store $(le VALSTORE disk) --workers ${WORKERS:-6} --sampler-slots ${SLOTS:-4}
  --log-every ${LOGE:-10} --val-every ${VAL:-100} --val-batches ${VALB:-4} --ckpt-every ${CKPT:-100} --seed 3 --val-seed 1 --warmup ${WARM:-40}
  --muon-lr $(le MLR 0.003) --lr $(le ALR 0.0003)
  --zfix 1 --symmetry-p 0.5 --geometry-aug 1 --rotate-deg 5 --rotate-p 0.2 --elastic-p 0.15 --axis-jitter 1250 --axis-jitter-p 0.5 --axis-jitter-deg 20
  --aug none --aug-strength 1 --aug-blur-p 0.7 --aug-blur-s 1.6 --aug-lowres-p 0.1 --aug-lowres-s 0.6 --aug-paganin-p 0.2 --aug-paganin-s 1.0
  --aug-tone-p 0.3 --aug-gin-p 0.15 --aug-noise-p 0.15 --aug-cnoise-p 0.1 --aug-sharpen-p 0.03 --aug-ring-p 0.1 --aug-slice-p 0.1 --aug-quant-p 0.1 --aug-bias-p 0.15 --aug-cupping-p 0.1
  --lat 4 --lat-basis /vesuvius/ufsm/v2/lat/codec/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --um-rungs 4.8,9.6:2,19.2 --rung-shift-p 0.3
  --film 2 --fstem 16 --fstem-drop 0.25 --uw $(le UW 0) --grad-cos 500 --ink-head 1 --ink-mil 1 --ink-dice 0 --ink-air 8 --fib-head 1 --ink-smooth 0.1 --ink-cache-mb 4096
  --mix-every 250 --onfly-slots $(le ONFLYSLOTS 1) --onfly-mb $(le ONFLYMB 768) --strict-sources 1 "${OPTX[@]}" "${XFLAGS[@]}"
  --snap-every ${CKPT:-100} --snap-keep-every ${CKPT:-100} --snap-keep-last 2)
# (deliberate differences from production: P, steps, warmup, log / val / ckpt / snap cadence, sampler slots / workers, one GPU without --split, the sources)

# ---- argv-sync: our flags vs the last production trainer argv (flags we override on purpose are not compared)
argvsync() {
  local pl; pl=$(grep -aE 'trainer pid [0-9]+ argv: .*ufsm[^ ]* train ' "$PRODLOG" 2>/dev/null | tail -1)
  [ -n "$pl" ] || { row argv-sync WARN "no production trainer argv in $PRODLOG" "$PRODLOG"; return; }
  printf '%s\n' "${pl#*argv: }" > "$LG/argv-prod.txt"; printf '%s ' "$U" train SRC "${TARGS[@]}" > "$LG/argv-e2e.txt"; echo >> "$LG/argv-e2e.txt"
  local d; d=$(python3 -I - "$LG/argv-prod.txt" "$LG/argv-e2e.txt" <<'PY'
import sys
SKIP = {'--out', '--P', '--steps', '--warmup', '--log-every', '--val-every', '--val-batches', '--ckpt-every', '--workers', '--sampler-slots',
        '--snap-every', '--snap-keep-every', '--snap-keep-last', '--gpus', '--fibre-gpu', '--ink-gpu', '--resume', '--split', '--split-top', '--stop-at'}
def flags(path):
    t = open(path).read().split(); t = t[t.index('train') + 2:]   # (binary train SOURCES ...)
    f, k = {}, None
    for w in t:
        if w.startswith('--'): k = w; f[k] = []
        elif k: f[k].append(w)
    return {a: ' '.join(b) for a, b in f.items() if a not in SKIP}
p, e = flags(sys.argv[1]), flags(sys.argv[2])
out = [f'missing {a} {p[a]}' for a in sorted(p) if a not in e] + [f'extra {a} {e[a]}' for a in sorted(e) if a not in p] + \
      [f'{a}: prod {p[a]} e2e {e[a]}' for a in sorted(p) if a in e and p[a] != e[a]]
print('; '.join(out))
PY
)
  if [ -z "$d" ]; then row argv-sync PASS "same flags as the last production argv (overrides excluded)" "$LG/argv-prod.txt"
  else row argv-sync WARN "$d" "$LG/argv-prod.txt"; fi
}
argvsync

# ---- 1 / 2: train + resume
TR=$O/train; SJ=$O/sources-e2e.json
mksources() {
  python3 -I - "$SRCJ" "$SJ" "$TRAINBOX" $E2E_SOURCES <<'PY'
import json, sys
src, dst, box, names = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(',')], sys.argv[4:]
d = json.load(open(src)); keep = [s for s in d['sources'] if s['name'] in names]
missing = set(names) - {s['name'] for s in keep}
if missing: sys.exit(f'e2e sources: not in {src}: {sorted(missing)}')
for s in keep: s['boxes'] = [box]   # (level-0 voxels: training windows inside it; holdout_file / holdout_mode kept)
d['sources'] = keep; d['note'] = f'e2e.sh: {src} reduced to {names}, windows inside level-0 box {box}'
json.dump(d, open(dst, 'w'), indent=1)
PY
}
loss_series() { awk '/^step +[0-9]+ / { for (i = 3; i < NF; i++) if ($i == "loss") { print $2, $(i + 1); break } }' "$1"; }
nanscan() { awk 'tolower($0) ~ /^(step|val)/ && tolower($0) ~ /(^|[^a-z])-?(nan|inf)([^a-z]|$)/ { n++ } END { print n + 0 }' "$1"; }
if want train; then
  mkdir -p "$TR"
  if ! mksources > "$LG/sources.log" 2>&1; then row train FAIL "sources json: $(tail -1 "$LG/sources.log")" "$LG/sources.log"
  else
    t0=$(date +%s); X "$LG/train1.log" ${TRAINTMO:-5400} env "${TRAIN_ENV[@]}" "$U" train "$SJ" --out "$TR" "${TARGS[@]}" --stop-at $STEPS1; rc=$?
    ls=$(loss_series "$LG/train1.log"); n=$(echo "$ls" | grep -c .); last=$(echo "$ls" | tail -1 | awk '{print $1 + 0}')
    f3=$(echo "$ls" | head -3 | awk '{s += $2; n++} END {if (n) printf "%.4f", s / n}'); l3=$(echo "$ls" | tail -3 | awk '{s += $2; n++} END {if (n) printf "%.4f", s / n}')
    nn=$(nanscan "$LG/train1.log"); cs=$(ckstep "$TR/last.ckpt"); why=""
    [ $rc = 0 ] || why+=" rc $rc;"; [ "${last:-0}" -ge $STEPS1 ] || why+=" last logged step ${last:-none} < $STEPS1;"; [ "$nn" = 0 ] || why+=" $nn nan/inf lines;"
    [ "${cs:-x}" = $STEPS1 ] || why+=" last.ckpt step ${cs:-missing} != $STEPS1;"
    [ "$n" -ge 6 ] && [ -n "$f3" ] && awkc "$l3 < ${LOSSDROP:-0.95} * $f3" || why+=" loss did not fall ($f3 -> $l3, need < ${LOSSDROP:-0.95} x);"
    met train_loss_first3 "$f3"; met train_loss_last3 "$l3"; met train_s $(( $(date +%s) - t0 ))
    nums="loss first3 $f3 -> last3 $l3 ($n logged), step $last, ckpt step ${cs:-none}, nan $nn, $(( $(date +%s) - t0 )) s, snaps $(ls "$TR/snap" 2>/dev/null | grep -c ckpt)"
    [ -z "$why" ] && row train PASS "$nums" "$LG/train1.log" || row train FAIL "$nums;$why" "$LG/train1.log"
  fi
  if ok_of train; then
    t0=$(date +%s); X "$LG/train2.log" ${TRAINTMO:-5400} env "${TRAIN_ENV[@]}" "$U" train "$SJ" --out "$TR" "${TARGS[@]}" --resume "$TR/last.ckpt"; rc=$?
    ls2=$(loss_series "$LG/train2.log"); first=$(echo "$ls2" | head -1 | awk '{print $1 + 0}'); last=$(echo "$ls2" | tail -1 | awk '{print $1 + 0}')
    m2=$(echo "$ls2" | awk '{s += $2; n++} END {if (n) printf "%.4f", s / n}'); nn=$(nanscan "$LG/train2.log"); cs=$(ckstep "$TR/last.ckpt"); why=""
    [ $rc = 0 ] || why+=" rc $rc$(grep -m1 -oE -- '--resume .*(do not carry over|refus)[^;]*' "$LG/train2.log" | sed 's/^/: /');"
    [ "${first:-0}" -gt $STEPS1 ] || why+=" first step ${first:-none} not after $STEPS1 (restarted?);"; [ "${last:-0}" -ge $STEPS ] || why+=" last step ${last:-none} < $STEPS;"
    [ "$nn" = 0 ] || why+=" $nn nan/inf lines;"; [ "${cs:-x}" = $STEPS ] || why+=" last.ckpt step ${cs:-missing} != $STEPS;"
    [ -n "$m2" ] && awkc "$m2 <= ${RESUMEFAC:-1.3} * $l3" || why+=" resumed loss $m2 > ${RESUMEFAC:-1.3} x run 1's last3 $l3;"
    met resume_loss_mean "$m2"
    nums="steps ${first:-?}..${last:-?}, loss mean $m2 (run 1 last3 $l3), ckpt step ${cs:-none}, $(( $(date +%s) - t0 )) s"
    [ -z "$why" ] && row resume PASS "$nums" "$LG/train2.log" || row resume FAIL "$nums;$why" "$LG/train2.log"
  else row resume SKIP "train failed" "$LG/train1.log"; fi
  if [ "${CLEAN:-1}" = 1 ]; then for d in "$TR/ring" "$TR/snap"; do [ -e "$d" ] && saferm "$d"; done; fi
fi

# ---- predict / fit helpers (the wbench recipe: cube window 640 / halo 64 for latent-grid nets, 19.2 um cells, lossless phase)
pfields() {   # pfields CKPT BOX FIELDS LOG EMA: fit fields of BOX
  local ck=$1 bx=$2 f=$3 lg=$4 em=$5; [ -e "$f" ] && saferm "$f"; [ -e "$f.tmp" ] && saferm "$f.tmp"
  X "$lg" ${PREDTMO:-1800} "$U" predict "$ck" "$CTR" "$CTK" "$f.tmp" --box "$bx" --fit-zarr "$f" --um $CTUM --level $LV --window 640 --halo 64 --shard 512 \
    --grid-origin 0,0,0 --ema $em --q 8 --threads $TH --gpu 0 --axis "$AX" --levels 1 --fit-h 2 --fit-q 8 --fit-q-phase 0 --no-out 1 $(gnflag "$ck"); local rc=$?
  [ -e "$f.tmp" ] && saferm "$f.tmp"; [ $rc = 0 ] && [ -e "$f/fields.json" ]; }
pheads() {   # pheads CKPT BOX OUTDIR LOG EMA: every head (the ueval recipe)
  local ck=$1 bx=$2 h=$3 lg=$4 em=$5; [ -e "$h" ] && saferm "$h"
  X "$lg" ${PREDTMO:-1800} "$U" predict "$ck" "$CTR" "$CTK" "$h" --um $CTUM --level $LV --box "$bx" --full-grid 1 --levels 1 --heads all --ema $em \
    --gpu 0 --axis "$AX" --threads 6 $(gnflag "$ck") && [ -n "$(ls "$h" 2>/dev/null)" ]; }
boxfit() {   # boxfit FIELDS OUT LOG: the production box fit, scored against the GT codes (wbench fitsc)
  X "$3" ${FITTMO:-1800} env UFSM_FIT_THREADS=$TH "$U" fit "$1" --gpu 0 --out "$2" --codes "$QR" $QK --dir 1 --uw-repair 8 --score-pairs 1; }
g_ids() { num "$(grep -oE 'winding ids right [0-9.]+' "$1" 2>/dev/null | tail -1 | awk '{print $4}')"; }
g_pair() { num "$(grep -oE 'pair winding right.*' "$1" 2>/dev/null | tail -1 | grep -oE "$2 mm [0-9.na]+ cross [0-9.na]+" | awk '{print $5}')"; }   # cross-sheet pairs
refrow() {   # refrow COL: the reference wbench row's value (REFTAG, HOLD)
  awk -F'\t' -v c="$1" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == c) k = i; next } $2 == "HOLD" && k { v = $k } END { print v }' "$WB/res/$REFTAG.tsv" 2>/dev/null; }

# ---- 3: the e2e net on HOLD (smoke, not gated on quality)
E2ECK=$TR/last.ckpt
if want e2e-pred; then
  d=$O/hold-e2e; mkdir -p "$d"
  if [ ! -e "$E2ECK" ]; then row e2e-pred SKIP "no e2e checkpoint ($E2ECK)" "$LG/train1.log"
  elif ! pfields "$E2ECK" $HOLDBOX "$d/fields" "$LG/e2e-fields.log" 0; then row e2e-pred FAIL "fit-field predict failed" "$LG/e2e-fields.log"
  elif ! pheads "$E2ECK" $HOLDBOX "$d/h" "$LG/e2e-heads.log" 0; then row e2e-pred FAIL "heads predict failed" "$LG/e2e-heads.log"
  elif ! boxfit "$d/fields" "$d/fit" "$LG/e2e-fit.log"; then row e2e-pred FAIL "box fit failed" "$LG/e2e-fit.log"
  else i=$(g_ids "$LG/e2e-fit.log"); met e2e_hold_ids0 "$i"
    [ "$i" != na ] && row e2e-pred PASS "HOLD ids0 $i p1x $(g_pair "$LG/e2e-fit.log" 1) (info only: a $STEPS-step net), heads: $(ls "$d/h" | tr '\n' ' ')" "$LG/e2e-fit.log" ||
      row e2e-pred FAIL "no GT score in the fit log" "$LG/e2e-fit.log"; fi
fi

# ---- 4: the reference snapshot on HOLD vs its wbench row
if want ref-box; then
  d=$O/hold-ref; mkdir -p "$d"
  r_ids=$(num "$(refrow ids0)"); r_p05=$(num "$(refrow p05x)"); r_p1=$(num "$(refrow p1x)")
  [ "$r_ids" = na ] && { r_ids=0.8095; r_p05=0.8818; r_p1=0.8333; log "ref-box: no $REFTAG HOLD row in $WB/res: the 11-14 values"; }
  if ! pfields "$REFCK" $HOLDBOX "$d/fields" "$LG/ref-fields.log" 1; then row ref-box FAIL "fit-field predict failed" "$LG/ref-fields.log"
  elif ! boxfit "$d/fields" "$d/fit" "$LG/ref-fit.log"; then row ref-box FAIL "box fit failed" "$LG/ref-fit.log"
  else i=$(g_ids "$LG/ref-fit.log"); p05=$(g_pair "$LG/ref-fit.log" 0.5); p1=$(g_pair "$LG/ref-fit.log" 1); why=""
    met ref_hold_ids0 "$i"; met ref_hold_p05x "$p05"; met ref_hold_p1x "$p1"
    [ "$i" != na ] && awkc "$i >= $r_ids - ${IDSTOL:-0.05}" || why+=" ids0 below $r_ids - ${IDSTOL:-0.05};"
    [ "$p05" != na ] && awkc "$p05 >= $r_p05 - ${PTOL:-0.03}" || why+=" p05x below $r_p05 - ${PTOL:-0.03};"
    [ "$p1" != na ] && awkc "$p1 >= $r_p1 - ${PTOL:-0.03}" || why+=" p1x below $r_p1 - ${PTOL:-0.03};"
    nums="ids0 $i (ref $r_ids) p05x $p05 (ref $r_p05) p1x $p1 (ref $r_p1), $REFTAG"
    [ -z "$why" ] && row ref-box PASS "$nums" "$LG/ref-fit.log" || row ref-box FAIL "$nums;$why" "$LG/ref-fit.log"; fi
fi

# ---- 5..11: the unroll chain on CHAINBOX
base() { awk -F'\t' -v k="$1" '$1 == k { v = $2 } END { print v }' "$BASE" 2>/dev/null; }
vsbase() {   # vsbase KEY VALUE MODE TOL: "" when fine, else the reason (MODE hi: VALUE >= base - TOL; rel: |VALUE / base - 1| <= TOL)
  local b; b=$(num "$(base "$1")"); [ "$b" = na ] && return 0; [ "$2" = na ] && { echo " $1 na (baseline $b);"; return 0; }
  case $3 in hi) awkc "$2 >= $b - $4" || echo " $1 $2 < baseline $b - $4;";;
             rel) awkc "$b == 0 ? $2 == 0 : ($2 / $b - 1 <= $4 && 1 - $2 / $b <= $4)" || echo " $1 $2 vs baseline $b beyond +-$4 relative;";; esac; }
if want chain; then
  C=$O/chain; F=$C/fields; REF=$C/ref; mkdir -p "$C"
  case ${CHAINCK:-ref} in e2e) CCK=$E2ECK; CEMA=0;; *) CCK=$REFCK; CEMA=1;; esac
  [ -s "$BASE" ] && log "chain: baseline $BASE" || log "chain: no baseline $BASE yet: absolute floors only (BLESS=1 on a good run writes it)"
  if ! pfields "$CCK" $CHAINBOX "$F" "$LG/chain-fields.log" $CEMA; then row chain-predict FAIL "fit-field predict failed" "$LG/chain-fields.log"
  elif ! pheads "$CCK" $CHAINBOX "$C/h" "$LG/chain-heads.log" $CEMA; then row chain-predict FAIL "heads predict failed" "$LG/chain-heads.log"
  else row chain-predict PASS "fields $(jq -c .shape "$F/fields.json") (h 2), heads: $(ls "$C/h" | tr '\n' ' ')($(basename "$CCK"))" "$LG/chain-fields.log"; fi

  # count-field reference (unroll.sh ref(): refpool flags, --dir from spiral_dir.tsv)
  sd=$(awk -F'\t' -v s=$SCROLL '$1 == s && ($2 == "1" || $2 == "+1" || $2 == "-1") { print $2 }' "$H/scripts/unroll/spiral_dir.tsv" | head -1)
  if ! ok_of chain-predict; then row count-field SKIP "no fields" "$LG/chain-fields.log"
  elif [ -z "$sd" ]; then row count-field FAIL "no spiral dir for $SCROLL in spiral_dir.tsv" "$H/scripts/unroll/spiral_dir.tsv"
  else [ -e "$REF" ] && saferm "$REF"
    if X "$LG/ref.log" ${CFTMO:-3600} "$U" count-field "$F" "$C/cf.unused" --R 0 --pool 8 --phase-w 0.5 --wide 100,140 --wide-phase 15 --model-only 1 --model-stride 1 \
         --dir $sd --iters 1200 --model "$REF" --threads $TH --mem-gb ${CFMEM:-8} && [ -e "$REF/model.json" ]; then
      rd=$(jq -r .dir "$REF/model.json"); met cf_dir "$rd"
      [ "$rd" = "${sd#+}" ] && row count-field PASS "dir $rd (spiral_dir.tsv $sd); $(grep -oE 'count-field: dir [+-]1 \([^)]*\)' "$LG/ref.log" | tail -1)" "$LG/ref.log" ||
        row count-field FAIL "model dir $rd != spiral_dir.tsv $sd" "$LG/ref.log"
    else row count-field FAIL "count-field failed: $(grep -o 'REFUSED.*' "$LG/ref.log" | tail -1 | cut -c1-160)" "$LG/ref.log"; fi
    [ -e "$C/cf.unused" ] && saferm "$C/cf.unused"
  fi

  # slab fits (unroll.sh lane(): SLABI 64 / SLABOV 16 production; here 2 slabs of 128 interior cells, overlap 16, snapped to the ref)
  if ok_of count-field; then
    ze=$(jq -r '.shape[0]' "$F/fields.json"); ny=$(jq -r '.shape[1]' "$F/fields.json"); nx=$(jq -r '.shape[2]' "$F/fields.json"); I=${SLABI:-$(( (ze + 1) / 2 ))}; OV=${SLABOV:-16}
    JARGS=(); nf=0; ns=0; sl=""
    for k in $(seq 0 $(( (ze + I - 1) / I - 1 ))); do
      s=$(printf 's%03d' $k); sdir=$C/slabs/$s; [ -e "$sdir" ] && saferm "$sdir"; mkdir -p "$sdir"
      c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
      bash "$H/scripts/wbench/mkcrop.sh" "$F" "$sdir/f" $a 0 0 $((b - a)) $ny $nx > "$LG/slab-$s.crop.log" 2>&1
      if X "$LG/slab-$s.log" ${FITTMO:-1800} env UFSM_FIT_THREADS=$TH "$U" fit "$sdir/f" --gpu 0 --out "$sdir/fit" --dir $rd --ref-model "$REF" --ref-snap 1 --uw-repair 0 \
           --crop-material ${CROPM:-2} && ! grep -qE 'illegal memory|loss 0\.00000' "$LG/slab-$s.log" && grep -qE '^fit: .*loss [0-9]' "$LG/slab-$s.log"; then
        JARGS+=("$sdir/fit" $((2 * c0)) $((2 * c1))); sl+=" $s[$c0,$c1)"
      else nf=$((nf + 1)); sl+=" $s FAILED"; fi; ns=$((ns + 1))
    done
    [ $nf = 0 ] && row slab-fits PASS "$ns slabs:$sl" "$LG/slab-s000.log" || row slab-fits FAIL "$nf of $ns slab fits bad:$sl" "$LG/slab-s000.log"
  else row slab-fits SKIP "no ref" "$LG/ref.log"; fi

  # fit-join (unroll.sh join()), the joined model's GT score (wbench slab-site recipe), wrap-check (unroll.sh check())
  if ok_of slab-fits; then [ -e "$C/joined" ] && saferm "$C/joined"
    if X "$LG/join.log" ${CFTMO:-3600} "$U" fit-join "$C/joined" "$F" --blend 16 --ref "$REF" "${JARGS[@]}" &&
       X "$LG/joined-score.log" ${FITTMO:-1800} env UFSM_FIT_THREADS=$TH "$U" fit "$F" --model "$C/joined" --gpu 0 --codes "$QR" $QK --score-pairs 1; then
      jok=1; ag=$(num "$(grep 'integer agreement on the overlaps' "$LG/join.log" | tail -1 | awk '{print $NF}')"); ji=$(g_ids "$LG/joined-score.log"); jp=$(g_pair "$LG/joined-score.log" 1)
      met join_agree "$ag"; met joined_ids "$ji"; met joined_p1x "$jp"
      X "$LG/wrapcheck.log" ${CFTMO:-3600} "$U" wrap-check "$C/joined" "$F" --ct "$CTR" "$CTK/$LV" --ct-um $LUM --tsv "$C/wrapcheck.tsv" --threads $TH; wrc=$?
      wv=$(grep -E '^wrap-check: verdicts' "$LG/wrapcheck.log" | tail -1 | sed 's/^wrap-check: //'); nr=$(echo "$wv" | grep -oE '[0-9]+ real' | awk '{print $1}')
      emp=$(awk -F'\t' 'NR > 1 { n++; e += $3 } END { if (n) printf "%.3f", e / n; else print "na" }' "$C/wrapcheck.tsv" 2>/dev/null); met wrapcheck_real "$(num "$nr")"
      why=""; [ "$ji" = na ] && why+=" no joined ids;"
      # (no absolute floor by default: the 0.40 guess failed a good join, 11-16: llvm d7910396 0.340, mcc 9f63cdd / e52bb39 0.342; BASE + BTOL gate)
      [ -n "${MINJOINED:-}" ] && [ "$ji" != na ] && ! awkc "$ji >= $MINJOINED" && why+=" joined ids $ji < MINJOINED $MINJOINED;"
      [ $wrc = 0 ] && [ "$emp" != na ] && awkc "$emp < 0.999" || why+=" wrap-check rc $wrc / no CT peaks (empty share $emp);"
      why+=$(vsbase joined_ids "$ji" hi ${BTOL:-0.05})$(vsbase joined_p1x "$jp" hi ${BTOL:-0.05})$(vsbase join_agree "$ag" hi ${BTOL:-0.05})
      nums="GT joined ids $ji p1x $jp, overlap agreement $ag, wrap-check $wv (CT-empty share $emp)"
      [ -z "$why" ] && row join PASS "$nums" "$LG/joined-score.log" || row join FAIL "$nums;$why" "$LG/joined-score.log"
    else row join FAIL "fit-join or its GT score failed" "$LG/join.log"; fi
  else row join SKIP "slab fits failed" "$LG/slab-s000.log"; fi

  # wraps (unroll.sh wraps(); --min-cm2 small: a box, not a scroll). Each downstream step runs whenever its input exists and gates only
  # its own output: a metric miss on join must not hide a broken wraps / flatten / render (11-16: join FAIL on the guessed floor
  # skipped the rest of the chain).
  if [ "${jok:-0}" = 1 ] && [ -e "$C/joined/set.json" ]; then [ -e "$C/wraps" ] && saferm "$C/wraps"
    if X "$LG/wraps.log" ${CFTMO:-3600} "$U" wraps "$C/joined" "$C/wraps" --step 4 --out-um $LUM --sfc 0.25 --fields "$F" --threads $TH --min-cm2 ${MINCM2W:-0.05}; then
      nw=$(ls "$C/wraps" 2>/dev/null | grep -c '^w.*\.sfc$'); met wraps_n "$nw"; why=""
      [ "$nw" -ge ${MINWRAPS:-3} ] || why+=" $nw wraps < MINWRAPS ${MINWRAPS:-3};"; why+=$(vsbase wraps_n "$nw" rel 0.3)
      grep -q 'wraps: WARNING: no-data' "$LG/wraps.log" && why+=" $(grep -o 'no-data share.*' "$LG/wraps.log" | tail -1);"
      [ -z "$why" ] && row wraps PASS "$nw wraps ($(ls "$C/wraps" | grep '\.sfc$' | head -1)..$(ls "$C/wraps" | grep '\.sfc$' | tail -1))" "$LG/wraps.log" ||
        row wraps FAIL "$nw wraps;$why" "$LG/wraps.log"
    else row wraps FAIL "wraps failed" "$LG/wraps.log"; fi
  else row wraps SKIP "no joined model" "$LG/join.log"; fi

  # flatten + render the largest wrap (unroll.sh pw_one recipe): on any wraps output, whatever the wraps row's count gates said
  if ls "$C/wraps"/w*.sfc > /dev/null 2>&1 && ! grep -qP '^wraps\t(SKIP|FAIL\twraps failed)' "$RES"; then
    w=$(for f in "$C/wraps"/w*.sfc; do echo "$(stat -c %s "$f") $(basename "$f" .sfc)"; done | sort -rn | head -1 | awk '{print $2}')
    nv=$(stat -c %s "$C/wraps/$w.sfc"); dc=$(awk -v n=$nv -v m=1500000 'BEGIN { d = 1; while (n / (d * d) > m) d++; print d }')
    for d in orig flat up r; do [ -e "$C/$d" ] && saferm "$C/$d"; done; mkdir -p "$C/r/in"
    if X "$LG/flat-scale.log" 600 "$U" tifxyz-scale "$C/wraps" "$C/orig" 1 --wraps $w &&
       X "$LG/flatten.log" ${FITTMO:-1800} "$U" flatten "$C/orig" "$C/flat" --wraps $w --decim $dc --threads ${FTH:-4} --crop none --gpu 1 --min-cm2 ${MINCM2F:-0.1} &&
       [ -e "$C/flat/$w/x.tif" ]; then
      cm=$(num "$(grep -E '^[^ ]+: input' "$LG/flatten.log" | head -1 | grep -oE '= [0-9.]+ cm2' | awk '{print $2}')"); met flat_cm2 "$cm"
      why=$(vsbase flat_cm2 "$cm" rel 0.3)
      [ -z "$why" ] && row flatten PASS "$w: $cm cm2 (decim $dc)" "$LG/flatten.log" || row flatten FAIL "$w: $cm cm2;$why" "$LG/flatten.log"
    else row flatten FAIL "flatten of $w failed" "$LG/flatten.log"; fi
  else row flatten SKIP "no wraps" "$LG/wraps.log"; fi
  if [ -n "${w:-}" ] && [ -e "$C/flat/$w/x.tif" ] && ! grep -qP '^flatten\t(SKIP|FAIL\tflatten of)' "$RES"; then
    ik=$(ls "$C/h/ink" 2>/dev/null | grep -v zarr.json | head -1)
    if [ -z "$ik" ]; then row render FAIL "no ink head output in $C/h" "$LG/chain-heads.log"
    elif X "$LG/flat-up.log" 600 "$U" tifxyz-scale "$C/flat" "$C/up" 1 --wraps $w --flip-rows 1 && ln -sfn "$C/up/$w" "$C/r/in/$w" &&
         X "$LG/render.log" ${FITTMO:-1800} env UFSM_RENDER_MEM=16 UFSM_RENDER_CHUNKED=1 UFSM_RENDER_GPU_MEM=4 "$U" render "$C/r/in" "$C/r" --volume "$C/h/ink" "$ik" \
           --ct "$CTR" "$CTK/$LV" --depth 1 --scale 1 --threads 8 &&
         [ -s "$C/r/$w.tif" ] && [ -s "$C/r/${w}_ct.tif" ] &&
         X "$LG/inkcompare.log" 600 "$U" ink-compare "$C/r/${w}_side.jpg" --ct "$C/r/${w}_ct.tif" --ink "ink=$C/r/$w.tif" --px-um $LUM --stretch --stretch-ct --max-px 8000 --threads 8 &&
         [ -s "$C/r/${w}_side.jpg" ]; then
      row render PASS "$w ink + CT $(stat -c %s "$C/r/$w.tif") / $(stat -c %s "$C/r/${w}_ct.tif") B; $(grep -v '^+' "$LG/inkcompare.log" | tail -1 | cut -c1-160)" "$LG/render.log"
    else row render FAIL "render / ink-compare of $w failed" "$LG/render.log"; fi
  else row render SKIP "no flat wrap" "$LG/flatten.log"; fi
fi

# ---- table, baseline, exit status
{ printf 'step\tresult\tnumbers\tlog\n'; cat "$RES"; } > "$O/table.tsv"
echo; echo "e2e $BSHA  $O"; column -t -s $'\t' "$O/table.tsv" | cut -c1-${COLS:-400}
nbad=$(awk -F'\t' '$2 == "FAIL" || $2 == "SKIP"' "$RES" | wc -l)
if [ "${BLESS:-0}" = 1 ]; then
  # (a STEPS_ONLY run blesses only its own keys: they replace those of BASE, the other keys are kept)
  if [ "$nbad" = 0 ]; then mkdir -p "$(dirname "$BASE")"
    { printf '# e2e baseline: %s %s %s%s\n' "$(date +%F)" "$BSHA" "$O" "${STEPS_ONLY:+ (steps: $STEPS_ONLY)}"
      [ -n "${STEPS_ONLY:-}" ] && [ -s "$BASE" ] && awk -F'\t' 'NR == FNR { k[$1]; next } !/^#/ && !($1 in k)' "$MET" "$BASE"
      cat "$MET"; } > "$BASE.tmp" && mv "$BASE.tmp" "$BASE" && log "blessed: $BASE"
  else log "BLESS=1 ignored: $nbad failed / skipped rows"; fi
fi
[ "$nbad" = 0 ] && { log "e2e PASS"; exit 0; }
log "e2e FAIL: $nbad rows failed or skipped"; exit 1

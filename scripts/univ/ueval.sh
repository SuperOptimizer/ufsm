#!/usr/bin/env bash
# scripts/univ/ueval.sh CKPT TAG: the universal latent net's eval harness on the desk (2026-11-12). The net trains on every label we have
# (user: "train on literally everything we have labels for ... then working on unseen scrolls from the corpus"), so two blocks:
#  (1) unseen (the main block): label-free quality on corpus scrolls with no labels and no fits anywhere (scripts/univ/unseen_boxes.tsv: one
#      512^3 level-0 box on each of 8 such scans, 8.64 / 9.36 um, picked CPU-side by unseen_boxes.py). Per box one GPU piece: fit fields
#      (predict --fit-zarr, the wbench recipe at the scan's own um), every head (predict --heads all, --full-grid 1), the fit (fit --dir 1
#      --uw-repair 8, GPU), its wraps, CT + ink renders on the largest wrap (render is GPU-only). Then CPU-side (scripts/univ/uqa.py, the fibre
#      judge): sheets on papyrus / gaps in voids / CT contrast, spacing regularity, winding jumps, normal smoothness, rv side vs the phase,
#      ink off papyrus / on the sheets, fibre judge (the fib head's fibres vs the box's own winding fit: sheet changes per m of fibre), the
#      fit's own residuals; one row per box -> $E/TAG/unseen.tsv, PNGs per box (mid-z slice with the sheet band and ink; ink | CT on the
#      largest wrap). Paired against REFS' unseen rows by scripts/univ/upair.py (per metric: mean B - A, bootstrap CI over boxes, wins).
#  (2) fit (training-fit, labelled as such: convergence and head interference, not generalisation): the former held-out tasks, all now in
#      training: wind (wbench core, a piece per box, 44), rv (scripts/rv/eval.sh, a piece per scroll of RVSC), ink (the 34 crops, 2-pass
#      blend, a piece per crop), fib (the Paris 4 HOLD box via predict --heads fib, fiber-judge vs the phase GT).
# BLOCKS ("unseen fit"), TASKS (the fit block's: wind rv ink fib). GPUS: "0" | "0 1" (a worker per GPU over one queue) | SPLIT=1 (one worker
# on both GPUs: plain predicts with PX, default "--split z --gpus 0,1"; --heads predicts with PXH, empty until predict --heads takes --split).
# Locks: /tmp/gpuG-bench.lock per piece; a piece starts only while no other job holds or waits for that lock (inkB first; stopped waiters
# ignored) and the GPU holds no other process above 200 MiB (BUSYOK: a pgrep pattern of allowed co-tenants). DRY=1: the queue only.
# Reruns skip finished pieces ($E/TAG/q/*.done) and finished box QA. Env: B (the one binary; default build/ufsm), E ($WB/univ/ueval), REFS
# (tags to pair with), UB (the unseen box list), RVSC, BOXES (core), WIN / HALO (640 / 64), THREADS (12), CTROOT (/vesuvius/ufsm/v2/ct), UQAPAR (3).
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}
H=$(cd "$(dirname "$0")/../.." && pwd); B=${B:-$H/build/ufsm}; WB=${WB:-/vesuvius/ufsm/wbench}; E=${E:-$WB/univ/ueval}
BLOCKS=${BLOCKS:-unseen fit}; TASKS=${TASKS:-wind rv ink fib}; UB=${UB:-$H/scripts/univ/unseen_boxes.tsv}
RVSC=${RVSC:-p1667 p139 p4}; BOXES=${BOXES:-core}; REFS=${REFS:-}; GPUS=${GPUS:-0}; SPLIT=${SPLIT:-0}
PX=${PX-}; [ $SPLIT = 1 ] && PX=${PX:---split z --gpus 0,1}; PXH=${PXH:-}; CTROOT=${CTROOT:-/vesuvius/ufsm/v2/ct}; RVE=${RVE:-/mnt/scratch/rv/eval}
INKC=${INKC:-/vesuvius/ufsm/inkgt/evcrops /vesuvius/ufsm/inkgt/evcrops-ho}; INKGT=${INKGT:-/vesuvius/ufsm/inkgt/evgt /vesuvius/ufsm/inkgt/evgt-ho}
Q=$E/$T/q; LG=$E/$T.ueval.log; mkdir -p $Q
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LG; }
blk() { [[ " $BLOCKS " == *" $1 "* ]]; }
has() { blk fit && [[ " $TASKS " == *" $1 "* ]]; }
[ -x $B ] || { echo "ueval: no binary $B" >&2; exit 2; }
head -c 4096 $CK | grep -aq '"lat_step"' || log "warning: $CK has no lat_step (not a latent-grid checkpoint): the --heads predicts will refuse it"

# ---- the queue: one line per piece, longest first (unseen boxes, rv scrolls, fib, then wbench boxes and ink crops interleaved)
boxes() { if [ "$BOXES" = core ]; then grep -v '^#' $H/scripts/wbench/boxes.tsv | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1; else echo $BOXES | tr ' ' '\n'; fi; }
crops() { for c in $WB/inkeval/crops $INKC; do grep -v '^#' $c/crops.tsv | cut -f1; done; }
{ blk unseen && grep -v '^#' $UB | cut -f1 | sed 's/^/u:/'; has rv && for s in $RVSC; do echo rv:$s; done; has fib && echo fib:hold
  paste -d '\n' <(has wind && boxes | sed 's/^/wind:/') <(has ink && crops | sed 's/^/ink:/') | grep . ; } > $Q/pieces
log "ueval $T ckpt $CK binary $B ($(sha1sum $B | cut -c1-8)) blocks [$BLOCKS] fit tasks [$TASKS] gpus [$GPUS] split $SPLIT: $(wc -l < $Q/pieces) pieces"

# ---- the unseen block: a box's GPU piece (predicts, fit, wraps, renders) and its CPU QA
ubox() { IFS=$'\t' read -r un sc ur uk uum ulv uz uy ux unz uny unx uax _ <<< "$(awk -F'\t' -v b=$1 '$1 == b' $UB)"; [ -n "${unx:-}" ]; }
upiece() {   # upiece NAME g
  ubox $1 || return 1; local d=$E/$T/u/$1 bx=$uz,$uy,$ux,$unz,$uny,$unx; mkdir -p $d
  [ -e $d/fields/fields.json ] || { rm -rf $d/fields $d/f.tmp; $B predict $CK $ur $uk $d/f.tmp --box $bx --fit-zarr $d/fields --um $uum --level $ulv --window ${WIN:-640} \
      --halo ${HALO:-64} --shard 512 --grid-origin 0,0,0 --ema 1 --q 8 --threads ${THREADS:-12} --gpu $2 --axis $uax --levels 1 --fit-h 2 --fit-q 8 --fit-q-phase 0 $PX \
      --no-out 1 --cache $WB/cache > $d/fields.log 2>&1; local rc=$?; rm -rf $d/f.tmp; [ $rc = 0 ]; } || return 1
  [ -e $d/h.done ] || { rm -rf $d/h; $B predict $CK $ur $uk $d/h --um $uum --level $ulv --box $bx --full-grid 1 --levels 1 --heads all $PXH --gpu $2 --threads 6 \
      --cache $WB/cache > $d/heads.log 2>&1 && touch $d/h.done; } || return 1
  [ -e $d/fit/model.json ] || UFSM_FIT_THREADS=${THREADS:-12} $B fit $d/fields --gpu $2 --out $d/fit --dir 1 --uw-repair 8 > $d/fit.log 2>&1 || return 1
  [ -d $d/wr ] || $B wraps $d/fit $d/wr --fields $d/fields --min-cm2 0.05 > $d/wraps.log 2>&1
  local w; w=$(ls -S $d/wr/*/x.tif 2>/dev/null | head -1); [ -n "$w" ] || return 0; w=$(dirname $w); mkdir -p $d/rw/w; ln -sfn $w $d/rw/w/$(basename $w)
  local ik; ik=$(ls $d/h/ink 2>/dev/null | grep -v zarr.json | head -1)
  [ -n "$ik" ] && [ ! -s $d/rw/ink/$(basename $w).tif ] && $B render $d/rw/w $d/rw/ink --volume $d/h/ink $ik --depth 4 --threads 8 > $d/render_ink.log 2>&1
  [ -s $d/rw/ct/$(basename $w).tif ] || $B render $d/rw/w $d/rw/ct --volume $ur $uk/$ulv --depth 4 --threads 8 > $d/render_ct.log 2>&1
  return 0; }
uqa() {   # uqa NAME: CPU only, after the box's GPU piece -> $d/row.tsv
  ubox $1 || return 1; local d=$E/$T/u/$1 hk f; [ -e $d/fit/model.json ] || return 1
  $B fit-phase-box $d/fit $uz $uy $ux $unz $uny $unx $d/ph.raw --level 0 --step 2 > $d/fpb.log 2>&1 || return 1   # (the model's native voxels = level ulv)
  [ -s $d/ct.raw ] || $B read $ur $uk/$ulv $uz $uy $ux $unz $uny $unx $d/ct.raw --threads 8 > /dev/null 2>&1 || return 1
  local xa=(); for hk in rv ink; do f=$(ls $d/h/$hk 2>/dev/null | grep -v zarr.json | head -1); [ -n "$f" ] || continue
    { [ -s $d/$hk.raw ] || $B read $d/h/$hk $f $uz $uy $ux $unz $uny $unx $d/$hk.raw --threads 8 > /dev/null 2>&1; } && xa+=(--$hk $d/$hk.raw); done
  rm -f $d/uqa.tsv; python3 -I $H/scripts/univ/uqa.py $d $1 $uum $d/ph.raw $d/ct.raw $unz,$uny,$unx "${xa[@]}" --step 2 > $d/uqa.log 2>&1 || return 1
  # fibre judge: the fib head's H / V fibres (fiber-trace) against this box's own winding fit (a fibre should stay in one sheet)
  local fk pm=na; fk=$(ls $d/h/fib/H 2>/dev/null | grep -v zarr.json | head -1)
  if [ -n "$fk" ]; then for f in H V; do [ -s $d/fj_$f.tsv ] && continue
      $B fiber-trace - - $d/$f.fib --mask $d/h/fib/$f $fk --mask-origin 0,0,0 --box $uz,$uy,$ux,$unz,$uny,$unx --scaledown 2 --family $f --half 1024 --min-len 48 \
        --threads 4 $([ -d $d/h/fib/dyad ] && echo --dyad $d/h/fib/dyad) > $d/trace_$f.log 2>&1 &&
      $B fiber-judge $d/$f.fib --set $d/fit --um $uum --level 0 --slab $uz,$unz --region $uy,$((uy + uny)),$ux,$((ux + unx)) --tile 256 --d-mm 0.3,0.6,1.2,2.4 \
        --out $d/fj_$f.tsv > $d/judge_$f.log 2>&1; done
    pm=$(awk -F'\t' 'FNR == 1 { next } { a += $5 + $12; p += $7 + $14 } END { if (a > 0) printf "%.2f", p / a; else print "na" }' $d/fj_H.tsv $d/fj_V.tsv 2>/dev/null); fi
  # the fit's own residuals (no codes): the final stage's loss terms, the slip repair's L1 disagreement after its moves, seam pairs left
  local fl sl sm; fl=$(grep -E '^fit stage [0-9]+ .* step' $d/fit.log | tail -1 | grep -oE '(loss|phase|normal|density) [0-9.]+' | awk '{printf "%s\t", $2}')
  sl=$(grep -oE 'L1 disagreement [0-9]+ -> [0-9]+' $d/fit.log | tail -1 | awk '{print $5}'); sm=$(grep -oE 'seam pairs [0-9]+ -> [0-9]+' $d/fit.log | tail -1 | awk '{print $5}')
  [ -n "$fl" ] || fl=$'na\tna\tna\tna\t'
  { printf 'box\tscroll\tfit_loss\tfit_phase\tfit_normal\tfit_density\tslip_l1\tseams\tfib_chg_m\t'; head -1 $d/uqa.tsv | cut -f2-
    printf '%s\t%s\t%s%s\t%s\t%s\t' $1 $sc "$fl" "${sl:-na}" "${sm:-na}" "$pm"; tail -1 $d/uqa.tsv | cut -f2-; } > $d/row.tsv
  python3 -I $H/scripts/univ/uqa.py --wrap-png $d $1 >> $d/uqa.log 2>&1   # (ink | CT on the largest wrap, side by side)
  return 0; }

# ---- one piece on GPU G ("0,1" in split mode: the scripts get GPU 0 and the split flags)
run() {   # run PIECE G
  local k=${1%%:*} a=${1#*:} G=$2 g=$2; [ $SPLIT = 1 ] && g=0
  local IE=(env E=$E U=$B UP=$B GPU=$g THREADS=8 REFS= CTROOT=$CTROOT PXH="$PXH" INKC="$INKC" INKGT="$INKGT")
  case $k in
    u)    upiece $a $g;;
    wind) env SITE=desk WB=$WB GPU=$g SLABS=0 REP=${REP:-0} THREADS=${THREADS:-12} PRIO=eval UP=$B U=$B WZARGS= WIN=${WIN:-640} HALO=${HALO:-64} \
            LOCALCT=$CTROOT BOXES=$a PX="$PX" bash $H/scripts/wbench/wbench.sh $CK $T >> $E/$T/wbench.out 2>&1
          grep -qP "^$T\t$a\t" $WB/res/$T.tsv;;
    rv)   env E=$RVE GPU=$G U=$B LOCK= WZ= WINDOW=${WIN:-640} PX="$PX" bash $H/scripts/rv/eval.sh $CK $T $a >> $E/$T/rv_$a.out 2>&1
          grep -qE '^ +all' $RVE/$T/$a/ruler.log;;
    fib)  "${IE[@]}" TASKS=fib FIBP=heads bash $H/scripts/univ/eval.sh $CK $T >> $E/$T/fib.out 2>&1; ls $E/$T/fib/*.judge > /dev/null 2>&1;;
    ink)  "${IE[@]}" TASKS=ink INKP=heads LV_INK=1 BLEND=${BLEND:-1} INKONLY=$a bash $H/scripts/univ/eval.sh $CK $T >> $E/$T/ink.out 2>&1
          grep -qP "^$a\t" $E/$T/ink/vscores.tsv;;
  esac; }

# ---- GPU locks, per piece
waiter() { local p; for p in $(pgrep -f "^flock /tmp/gpu$1-bench.lock"); do [ "$(ps -o state= -p $p)" != T ] && return 0; done; return 1; }
busy() { local p m; while IFS=', ' read -r p m; do [ -n "$p" ] && [ "${m:-0}" -gt 200 ] || continue
    ps -o args= -p $p | grep -qE "${BUSYOK:-ufsm-fibd/}" || return 0; done < <(nvidia-smi --id=$1 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits); return 1; }
take() {   # take G...: every listed GPU's bench lock (fds 8, 9), each free of waiters and other processes
  local g fd=8; for g in "$@"; do while :; do while waiter $g || busy $g; do sleep 60; done
      eval "exec $fd> /tmp/gpu$g-bench.lock"; flock $fd; busy $g || break; flock -u $fd; eval "exec $fd>&-"; sleep 30; done; fd=$((fd + 1)); done; }
give() { flock -u 8 2>/dev/null; exec 8>&- 2>/dev/null; flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }

worker() {   # worker G: pull pieces until none is left
  local G=$1 p; while read -r p; do
    [ -e $Q/$p.done ] && continue; mkdir $Q/$p.claim 2> /dev/null || continue
    if [ "${DRY:-0}" = 1 ]; then echo "[gpu $G] $p"; continue; fi
    if [ $SPLIT = 1 ]; then take 0 1; else take $G; fi
    local t0=$(date +%s); run $p $G && touch $Q/$p.done && log "[gpu $G] $p done ($(( $(date +%s) - t0 )) s)" || log "[gpu $G] $p FAILED"
    give; rmdir $Q/$p.claim; done < $Q/pieces; }
rmdir $Q/*.claim 2> /dev/null   # (claims of a killed earlier run)
if [ $SPLIT = 1 ]; then worker 0,1; else for g in $GPUS; do worker $g & sleep ${STAGGER:-20}; done; wait; fi   # (stagger: the first wbench piece writes res/TAG.tsv's header)
[ "${DRY:-0}" = 1 ] && { rmdir $Q/*.claim 2> /dev/null; exit 0; }
left=$(while read -r p; do [ -e $Q/$p.done ] || echo $p; done < $Q/pieces | wc -l); log "pieces left: $left"

# ---- the unseen boxes' CPU QA (UQAPAR at a time), the unseen table
if blk unseen; then
  for n in $(grep -v '^#' $UB | cut -f1); do [ -e "$Q/u:$n.done" ] || continue; [ -s $E/$T/u/$n/row.tsv ] && continue
    while [ $(jobs -rp | wc -l) -ge ${UQAPAR:-3} ]; do sleep 2; done; (uqa $n || log "uqa $n FAILED") & done; wait
  { f=1; for n in $(grep -v '^#' $UB | cut -f1); do r=$E/$T/u/$n/row.tsv; [ -s $r ] || continue; [ $f = 1 ] && head -1 $r; tail -1 $r; f=0; done; } > $E/$T/unseen.tsv
fi

# ---- summary: $E/TAG.ueval.txt (CPU only)
{ echo "== ueval $T  $CK  $(date)"
  if blk unseen && [ -s $E/$T/unseen.tsv ]; then
    echo "-- UNSEEN scrolls (no labels anywhere; label-free self-consistency, $(($(wc -l < $E/$T/unseen.tsv) - 1)) boxes; images $E/$T/u/*/*.png)"
    column -t -s $'\t' $E/$T/unseen.tsv | sed 's/^/   /'
    for r in $REFS; do [ -s $E/$r/unseen.tsv ] && { echo "   paired vs $r (B - A over the boxes both have)"; python3 -I $H/scripts/univ/upair.py $E/$r/unseen.tsv $E/$T/unseen.tsv | sed 's/^/   /'; }; done
  fi
  blk fit && echo "== TRAINING-FIT block (every label below is in the net's training: convergence / head interference, not generalisation)"
  if has wind && [ -s $WB/res/$T.tsv ]; then echo "-- wind, training-fit: wbench core rows $(($(wc -l < $WB/res/$T.tsv) - 1))"
    for r in $REFS; do [ -s $WB/res/$r.tsv ] && { echo "   vs $r"; $B wbench-stats $WB/res/$r.tsv $WB/res/$T.tsv --metric p05x,ids 2>&1 | grep -E '^(p05x|ids) '; }; done
    [ -z "$REFS" ] && $B wbench-stats $WB/res/$T.tsv --metric p05x,ids 2>&1 | grep -E '^(p05x|ids) '; fi
  has rv && { echo "-- rv, training-fit (turns per GT turn / share of intervals off >= 0.5 turn)"; for s in $RVSC; do echo "   $s $(grep -E '^ +all' $RVE/$T/$s/ruler.log 2>/dev/null | tail -1)"; done; }
  if has ink && [ -s $E/$T/ink/vscores.tsv ]; then echo "-- ink, training-fit (3-D AUC / best F1, $([ ${BLEND:-1} = 1 ] && echo 2-pass blend || echo single pass); every crop's labels are in training)"
    awk -F'\t' -v t=$T '{ s = ($1 ~ /^w[0-9]/) ? "PHercParis4" : substr($1, 1, index($1, "_w") - 1); c[s]++; A[s] += $6; F[s] += $8; n++; AA += $6; FF += $8 }
      END { printf "%-16s  training-fit ink:", t; split("PHercParis4 PHerc0139 PHerc1667 PHerc0500P2 PHerc0841", o, " ")
        for (i = 1; i <= 5; i++) if (c[o[i]]) printf "  %s %.3f/%.3f", o[i], A[o[i]] / c[o[i]], F[o[i]] / c[o[i]]; printf "  all%d %.3f/%.3f\n", n, AA / n, FF / n }' \
      $E/$T/ink/vscores.tsv | tee $E/$T/ink-row.txt; fi
  has fib && { echo "-- fib, training-fit (fiber-judge vs the phase GT)"; for f in $E/$T/fib/*.judge; do [ -s $f ] && sed "s|^|   $(basename $f .judge) |" $f; done; }
} > $E/$T.ueval.txt
[ -s $E/$T/ink-row.txt ] && while IFS= read -r l; do grep -qxF "$l" $E/ink-rows.txt 2>/dev/null || echo "$l" >> $E/ink-rows.txt; done < $E/$T/ink-row.txt
cat $E/$T.ueval.txt | tee -a $LG

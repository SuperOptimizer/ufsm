#!/usr/bin/env bash
# scripts/univ/ueval.sh CKPT TAG: the universal latent net's eval harness on the desk (2026-11-12, the from-scratch universal net across both
# GPUs). One checkpoint, four tasks, cut into pieces that GPU workers pull from a shared queue, the GPU locks held per piece only:
#   wind  wbench core, one piece per box (44; stats on the 39 clean), REP (0) -> $WB/res/TAG.tsv, paired vs REFS
#   rv    recto / verso (scripts/rv/eval.sh: --channel rv on the ruler rows + recto-ruler), one piece per scroll of RVSC (p1667 p139 p4)
#   ink   the 34 ink crops (16 Paris 4 + inkB's 0139 / 1667 + held-out 0500P2 / 0841), predict --heads ink at 4.8 um with the 2-pass grid
#         blend (BLEND=1: --grid-origin 0 / 256, ink-blend --tile 512; the latent ink head's level steps at window seams, inkB), one piece
#         per crop; 3-D ink-vscore (AUC / best F1) + 2-D render / ink-score -> $E/TAG/ink, one row in $E/ink-rows.txt
#   fib   the Paris 4 HOLD box: predict --heads fib (FIBP=heads), fiber-trace tube / dyad, fiber-judge vs the phase GT -> $E/TAG/fib
# GPUS: "0" (one GPU), "0 1" (a worker per GPU, pieces in parallel), or SPLIT=1: one worker holding both GPUs, every plain predict (wbench
# fields, rv) with PX (default "--split z --gpus 0,1") and the --heads predicts with PXH (empty = GPU 0 alone: predict --heads takes no
# --split yet; set PXH once univI's split predict lands). Locks: /tmp/gpuG-bench.lock per piece (inkB and other jobs flock it); a piece starts
# only while no other job holds or waits for that lock (inkB first, lead 2026-11-12; a stopped waiter is ignored) and the GPU holds no other
# process above 200 MiB (BUSYOK: a pgrep pattern of allowed co-tenants). DRY=1: print the queue and the commands, run nothing.
# Reruns skip finished pieces ($E/TAG/q/*.done; wbench rows, rv ruler logs, ink scores, fib judges are each resumable themselves).
# Env: B (the binary for every predict / fit / score; default build/ufsm), E ($WB/univ/ueval), REFS (wbench-stats pairs, e.g. "unified-1504-ssl"),
# TASKS (wind rv ink fib), RVSC, BOXES (core), WIN / HALO (640 / 64), THREADS (12), CTROOT (/vesuvius/ufsm/v2/ct).
set -u
CK=$(readlink -f "${1:?ckpt}"); T=${2:?tag}
H=$(cd "$(dirname "$0")/../.." && pwd); B=${B:-$H/build/ufsm}; WB=${WB:-/vesuvius/ufsm/wbench}; E=${E:-$WB/univ/ueval}
TASKS=${TASKS:-wind rv ink fib}; RVSC=${RVSC:-p1667 p139 p4}; BOXES=${BOXES:-core}; REFS=${REFS:-}; GPUS=${GPUS:-0}; SPLIT=${SPLIT:-0}
PX=${PX-}; [ $SPLIT = 1 ] && PX=${PX:---split z --gpus 0,1}; PXH=${PXH:-}; CTROOT=${CTROOT:-/vesuvius/ufsm/v2/ct}; RVE=${RVE:-/mnt/scratch/rv/eval}
INKC=${INKC:-/vesuvius/ufsm/inkgt/evcrops /vesuvius/ufsm/inkgt/evcrops-ho}; INKGT=${INKGT:-/vesuvius/ufsm/inkgt/evgt /vesuvius/ufsm/inkgt/evgt-ho}
Q=$E/$T/q; LG=$E/$T.ueval.log; mkdir -p $Q
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LG; }
has() { [[ " $TASKS " == *" $1 "* ]]; }
[ -x $B ] || { echo "ueval: no binary $B" >&2; exit 2; }
head -c 4096 $CK | grep -aq '"lat_step"' || log "warning: $CK has no lat_step (not a latent-grid checkpoint): ink / fib --heads predicts will refuse it"

# ---- the queue: one line per piece, longest first (rv scrolls, fib, then boxes and crops interleaved so both GPUs finish together)
boxes() { if [ "$BOXES" = core ]; then grep -v '^#' $H/scripts/wbench/boxes.tsv | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1; else echo $BOXES | tr ' ' '\n'; fi; }
crops() { for c in $WB/inkeval/crops $INKC; do grep -v '^#' $c/crops.tsv | cut -f1; done; }
{ has rv && for s in $RVSC; do echo rv:$s; done; has fib && echo fib:hold
  paste -d '\n' <(has wind && boxes | sed 's/^/wind:/') <(has ink && crops | sed 's/^/ink:/') | grep . ; } > $Q/pieces
log "ueval $T ckpt $CK binary $B ($(sha1sum $B | cut -c1-8)) tasks [$TASKS] gpus [$GPUS] split $SPLIT: $(wc -l < $Q/pieces) pieces"

# ---- one piece on GPU G ("0,1" in split mode: the scripts get GPU 0 and the split flags)
run() {   # run PIECE G
  local k=${1%%:*} a=${1#*:} G=$2 g=$2; [ $SPLIT = 1 ] && g=0
  local IE=(env E=$E U=$B UP=$B GPU=$g THREADS=8 REFS= CTROOT=$CTROOT PXH="$PXH" INKC="$INKC" INKGT="$INKGT")
  case $k in
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

# ---- summary: $E/TAG.ueval.txt (CPU only; the summary pass of eval.sh builds the ink / fib matrix from the finished pieces)
{ echo "== ueval $T  $CK  $(date)"
  if has wind && [ -s $WB/res/$T.tsv ]; then echo "-- wind: wbench core rows $(($(wc -l < $WB/res/$T.tsv) - 1))"
    for r in $REFS; do [ -s $WB/res/$r.tsv ] && { echo "   vs $r"; $B wbench-stats $WB/res/$r.tsv $WB/res/$T.tsv --metric p05x,ids 2>&1 | grep -E '^(p05x|ids) '; }; done
    [ -z "$REFS" ] && $B wbench-stats $WB/res/$T.tsv --metric p05x,ids 2>&1 | grep -E '^(p05x|ids) '; fi
  has rv && { echo "-- rv (turns per GT turn / share of intervals off >= 0.5 turn)"; for s in $RVSC; do echo "   $s $(grep -E '^ +all' $RVE/$T/$s/ruler.log 2>/dev/null | tail -1)"; done; }
  if has ink && [ -s $E/$T/ink/vscores.tsv ]; then echo "-- ink (3-D AUC / best F1, $([ ${BLEND:-1} = 1 ] && echo 2-pass blend || echo single pass))"
    awk -F'\t' -v t=$T '{ s = ($1 ~ /^w[0-9]/) ? "PHercParis4" : substr($1, 1, index($1, "_w") - 1); c[s]++; A[s] += $6; F[s] += $8; n++; AA += $6; FF += $8 }
      END { printf "%-16s", t; split("PHercParis4 PHerc0139 PHerc1667 PHerc0500P2 PHerc0841", o, " "); for (i = 1; i <= 5; i++) if (c[o[i]]) printf "  %s %.3f/%.3f", o[i], A[o[i]] / c[o[i]], F[o[i]] / c[o[i]]; printf "  all%d %.3f/%.3f\n", n, AA / n, FF / n }' $E/$T/ink/vscores.tsv | tee $E/$T/ink-row.txt; fi
  has fib && { echo "-- fib (fiber-judge vs the phase GT)"; for f in $E/$T/fib/*.judge; do [ -s $f ] && sed "s|^|   $(basename $f .judge) |" $f; done; }
} > $E/$T.ueval.txt
[ -s $E/$T/ink-row.txt ] && { grep -qF "$(cat $E/$T/ink-row.txt)" $E/ink-rows.txt 2>/dev/null || cat $E/$T/ink-row.txt >> $E/ink-rows.txt; }
cat $E/$T.ueval.txt | tee -a $LG

#!/usr/bin/env bash
# scripts/univ/unifiedwatch.sh [FIRST]: score every 1000-step snapshot of the unified run as it lands (2026-11-12; latent net only).
# INK=0: no ink (winding-only runs); STEP (1000): the snapshot stride scored; BASECK "CKPT=TAG ...": wbench core rows of reference
# checkpoints scored once, with the same binary, before the run's first snapshot (the run is paired against them).
# RUN (default unified): the run's name, its snapshots SN=/vesuvius/ufsm/v2/lat/runs/RUN/snap, tags RUN-S (e.g. RUN=inkpod WEVERY=0: ink only).
# Watchers of several runs share GPU 0 one snapshot at a time (flock $E/gpu0.lock); $E/ink-rows.txt holds one row per (run, step).
#   ink    every snapshot: predict --heads ink on inkB's 34 crops (16 Paris 4 + 8 0139 / 1667 + 10 held-out 0500P2 / 0841) at 4.8 um,
#          voxelwise vs the 3-D ink-splat refs (ink-vscore) and 2-D on the crop surfaces (render + ink-score); per-scroll AUC / best F1
#          next to the dino3d teacher's existing rows (pod, the 24 crops; floor 0.513) -> $E/unified-S.matrix.txt + $E/unified-S.ink.txt
#   wind   every WEVERY (5000) steps: wbench core (44 boxes, REP=0: p05x) -> res/unified-S.tsv
# GPU 0 only (GPU 1 runs the unified trainer); the unified binary pinned as a copy (it knows its heads). One snapshot at a time.
set -u
H=${H:-$(cd "$(dirname "$0")/../.." && pwd)}; WB=/vesuvius/ufsm/wbench; RUN=${RUN:-unified}; SN=${SN:-/vesuvius/ufsm/v2/lat/runs/$RUN/snap}
B=${B:-/vesuvius/ufsm/v2/lat/latres/bin/ufsm.unified1}; E=${E:-$WB/univ/unified}; GPU=${GPU:-0}; WEVERY=${WEVERY:-5000}
TEACH=${TEACH:-$WB/univ/pod/ink3dext/vscores.tsv}; LG=$E/watch-$RUN.log; mkdir -p $E
# the unroll's GPU 0 jobs go first (lead 2026-11-12): univI's Paris 4 face renders (until sched/p4rv.done) and anything of the unroll
# schedule on GPU 0 (rvside / p1299 / unrollperf), then unrollperf2's window (sched/perf2.done), then univI's P4 column predicts (sched/p4col.done); checked after taking the lock
US=${US:-/vesuvius/ufsm/unroll/sched}
# and GPU 0 itself: no other compute process above 200 MiB (a marker written early or a renamed job; 2026-11-12 perf2.done came at the
# window's start) except allowed co-tenants (GPU0OK, a pgrep -f pattern: fibhvd's sampler check)
gpu0busy() { local p m; while IFS=', ' read -r p m; do [ -n "$p" ] && [ "${m:-0}" -gt 200 ] || continue
    ps -o args= -p $p | grep -qE "${GPU0OK:-ufsm-fibd/}" || return 0; done < <(nvidia-smi --id=${GPU:-0} --query-compute-apps=pid,used_memory --format=csv,noheader,nounits); return 1; }
perfopen() { local f; for f in $US/perf*.go; do [ -e "$f" ] && [ ! -e "${f%.go}.done" ] && return 0; done; return 1; }   # unrollperf2's windows: perfN.go without perfN.done
# bench lock (lead 2026-11-12): every GPU 0 job also holds /tmp/gpu0-bench.lock for its duration (inkB's unroll ink runs and others flock it),
# taken after the checks pass, then the GPU re-checked (someone may have started in between): gpulock / gpuunlock around each job
gpulock() { while :; do gpu0free; exec 8> ${BLK:-/tmp/gpu0-bench.lock}; flock 8; gpu0busy || return 0; flock -u 8; exec 8>&-; sleep 30; done; }
gpuunlock() { flock -u 8 2>/dev/null; exec 8>&- 2>/dev/null; }
# inkB first (lead 2026-11-12, run5): the locks are taken per ink crop / per wbench box and released in between; a piece starts only
# while no other process holds or waits for the bench lock (inkB's runner: flock /tmp/gpu0-bench.lock CMD)
yieldgpu() { while pgrep -f '^flock /tmp/gpu0-bench.lock' > /dev/null; do sleep 60; done; exec 9> $E/gpu0.lock; flock 9; gpulock; }
release() { gpuunlock; flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
gpu0free() { until [ -e $US/p4rv.done ] && [ -e $US/perf2.done ] && [ -e $US/p4col.done ] && ! pgrep -f "$US/(rvside|p4rv|p1299|unrollperf)|unrollperf" > /dev/null \
    && ! perfopen && ! gpu0busy && sleep 30 && ! gpu0busy; do sleep 60; done; }
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LG; }
perscroll() {   # perscroll VSCORES LABEL: per-scroll mean AUC / best F1 (crop names: w.. = Paris 4, else SCROLL_w..)
  awk -F'\t' -v L="$2" '{ s = ($1 ~ /^w[0-9]/) ? "PHercParis4" : substr($1, 1, index($1, "_w") - 1); c[s]++; A[s] += $6; F[s] += $8 }
    END { for (s in c) printf "%-12s %-10s n %2d auc %.3f f1best %.3f\n", s, L, c[s], A[s] / c[s], F[s] / c[s] }' $1 | sort; }
score() {   # score SNAPSHOT-STEP
  local s=$1 ck=$SN/$1.ckpt t=$RUN-$1
  if [ "${INK:-1}" = 1 ]; then
  log "$t: ink"
  local EV=(env E=$E U=$B UP=$B GPU=$GPU THREADS=8 TASKS=ink INKP=heads LV_INK=1 CTROOT=/vesuvius/ufsm/v2/ct REFS=
    INKC="/vesuvius/ufsm/inkgt/evcrops /vesuvius/ufsm/inkgt/evcrops-ho" INKGT="/vesuvius/ufsm/inkgt/evgt /vesuvius/ufsm/inkgt/evgt-ho") cr
  for cr in $(cut -f1 ${CROPLIST:-$E/inkpod-9000/ink/vscores.tsv}); do grep -qP "^$cr\t" $E/$t/ink/scores.tsv 2>/dev/null && continue
    yieldgpu; "${EV[@]}" INKONLY=$cr bash $H/scripts/univ/eval.sh $ck $t >> $E/$t.eval.out 2>&1; release; done
  yieldgpu; "${EV[@]}" bash $H/scripts/univ/eval.sh $ck $t >> $E/$t.eval.out 2>&1; release   # (the summary; locked in case a crop is retried)
  { echo "== $t ink (3-D: ink-vscore vs inkB's ink-splat refs, 4.8 um; held-out: PHerc0500P2 unseen ink labels / seen CT, PHerc0841 unseen scroll)"
    [ -s $E/$t/ink/vscores.tsv ] && perscroll $E/$t/ink/vscores.tsv $t
    [ -s $TEACH ] && perscroll $TEACH dino3d
    echo "(floor: ink share 0.513 AUC-equivalent per the lead; dino3d rows exist for the 24 crops only)"; } > $E/$t.ink.txt
  cat $E/$t.ink.txt >> $LG
  # one row per (run, step): 3-D AUC / best F1 per scroll, so runs line up at equal steps
  [ -s $E/$t/ink/vscores.tsv ] && awk -F'\t' -v t=$t '{ s = ($1 ~ /^w[0-9]/) ? "PHercParis4" : substr($1, 1, index($1, "_w") - 1); c[s]++; A[s] += $6; F[s] += $8; n++; AA += $6; FF += $8 }
    END { printf "%-16s", t; split("PHercParis4 PHerc0139 PHerc1667 PHerc0500P2 PHerc0841", o, " "); for (i = 1; i <= 5; i++) printf "  %s %.3f/%.3f", o[i], A[o[i]] / c[o[i]], F[o[i]] / c[o[i]]; printf "  all%d %.3f/%.3f\n", n, AA / n, FF / n }' $E/$t/ink/vscores.tsv >> $E/ink-rows.txt
  fi
  wcore $ck $t $s; }
wcore() {   # wcore CKPT TAG STEP: wbench core p05x rows every WEVERY steps (STEP 0: always)
  local ck=$1 t=$2 s=$3
  if [ $s = 0 ] || { [ $WEVERY -gt 0 ] && [ $((s % WEVERY)) = 0 ]; }; then log "$t: wbench core"
    local b; for b in $(grep -v '^#' $H/scripts/wbench/boxes.tsv | awk -F'\t' '$10 !~ /label-noise/ && ($2 == "paris4" || $2 == "paris4j" || $10 ~ /core/)' | cut -f1); do
      grep -qP "^$t\t$b\t" $WB/res/$t.tsv 2>/dev/null && continue; yieldgpu
      env SITE=desk WB=$WB GPU=$GPU SLABS=0 REP=0 THREADS=12 PRIO=eval UP=$B WZARGS= WIN=640 HALO=64 LOCALCT=/vesuvius/ufsm/v2/ct BOXES=$b \
        bash $H/scripts/wbench/wbench.sh $ck $t >> $E/$t.wbench.out 2>&1; release; done
    log "$t: wbench rows $(($(wc -l < $WB/res/$t.tsv) - 1))"; fi
}
S=${1:-1000}; STEP=${STEP:-1000}
for kv in ${BASECK:-}; do wcore ${kv%%=*} ${kv#*=} 0; done
while [ ! -e $E/watch.stop ]; do
  # the first snapshot at or after S (a run that starts its count elsewhere, e.g. from its init's step, is picked up too)
  until n=$(ls $SN 2>/dev/null | sed -n 's/^\([0-9]*\)\.ckpt$/\1/p' | awk -v s=$S '$1 >= s' | sort -n | head -1); [ -n "$n" ]; do [ -e $E/watch.stop ] && exit 0; sleep 60; done
  S=$n; sleep 60
  score $S; S=$((S + STEP))
done

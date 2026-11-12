#!/usr/bin/env bash
# scripts/univ/unifiedwatch.sh [FIRST]: score every 1000-step snapshot of the unified run as it lands (2026-11-12; latent net only).
#   ink    every snapshot: predict --heads ink on inkB's 34 crops (16 Paris 4 + 8 0139 / 1667 + 10 held-out 0500P2 / 0841) at 4.8 um,
#          voxelwise vs the 3-D ink-splat refs (ink-vscore) and 2-D on the crop surfaces (render + ink-score); per-scroll AUC / best F1
#          next to the dino3d teacher's existing rows (pod, the 24 crops; floor 0.513) -> $E/unified-S.matrix.txt + $E/unified-S.ink.txt
#   wind   every WEVERY (5000) steps: wbench core (44 boxes, REP=0: p05x) -> res/unified-S.tsv
# GPU 0 only (GPU 1 runs the unified trainer); the unified binary pinned as a copy (it knows its heads). One snapshot at a time.
set -u
H=${H:-$(cd "$(dirname "$0")/../.." && pwd)}; WB=/vesuvius/ufsm/wbench; SN=${SN:-/vesuvius/ufsm/v2/lat/runs/unified/snap}
B=${B:-/vesuvius/ufsm/v2/lat/latres/bin/ufsm.unified1}; E=${E:-$WB/univ/unified}; GPU=${GPU:-0}; WEVERY=${WEVERY:-5000}
TEACH=${TEACH:-$WB/univ/pod/ink3dext/vscores.tsv}; LG=$E/watch.log; mkdir -p $E
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LG; }
perscroll() {   # perscroll VSCORES LABEL: per-scroll mean AUC / best F1 (crop names: w.. = Paris 4, else SCROLL_w..)
  awk -F'\t' -v L="$2" '{ s = ($1 ~ /^w[0-9]/) ? "PHercParis4" : substr($1, 1, index($1, "_w") - 1); c[s]++; A[s] += $6; F[s] += $8 }
    END { for (s in c) printf "%-12s %-10s n %2d auc %.3f f1best %.3f\n", s, L, c[s], A[s] / c[s], F[s] / c[s] }' $1 | sort; }
score() {   # score SNAPSHOT-STEP
  local s=$1 ck=$SN/$1.ckpt t=unified-$1
  log "$t: ink"
  env E=$E U=$B UP=$B GPU=$GPU THREADS=8 TASKS=ink INKP=heads LV_INK=1 CTROOT=/vesuvius/ufsm/v2/ct REFS= \
    INKC="/vesuvius/ufsm/inkgt/evcrops /vesuvius/ufsm/inkgt/evcrops-ho" INKGT="/vesuvius/ufsm/inkgt/evgt /vesuvius/ufsm/inkgt/evgt-ho" \
    bash $H/scripts/univ/eval.sh $ck $t > $E/$t.eval.out 2>&1
  { echo "== $t ink (3-D: ink-vscore vs inkB's ink-splat refs, 4.8 um; held-out: PHerc0500P2 unseen ink labels / seen CT, PHerc0841 unseen scroll)"
    [ -s $E/$t/ink/vscores.tsv ] && perscroll $E/$t/ink/vscores.tsv $t
    [ -s $TEACH ] && perscroll $TEACH dino3d
    echo "(floor: ink share 0.513 AUC-equivalent per the lead; dino3d rows exist for the 24 crops only)"; } > $E/$t.ink.txt
  cat $E/$t.ink.txt >> $LG
  if [ $((s % WEVERY)) = 0 ]; then log "$t: wbench core"
    env SITE=desk WB=$WB GPU=$GPU SLABS=0 REP=0 THREADS=12 PRIO=eval UP=$B WZARGS= WIN=640 HALO=64 LOCALCT=/vesuvius/ufsm/v2/ct BOXES=core \
      bash $H/scripts/wbench/wbench.sh $ck $t > $E/$t.wbench.out 2>&1
    log "$t: wbench rows $(($(wc -l < $WB/res/$t.tsv) - 1))"; fi
}
S=${1:-1000}
while [ ! -e $E/watch.stop ]; do
  until [ -s $SN/$S.ckpt ]; do [ -e $E/watch.stop ] && exit 0; sleep 60; done; sleep 60
  score $S; S=$((S + 1000))
done

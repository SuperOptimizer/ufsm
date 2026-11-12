#!/usr/bin/env bash
# scripts/univ/u2ink_smoke.sh [ARM...]: the U2 ink smoke (2026-11-12): does the universal net's ink head learn the human 3-D ink GT?
# Ink-only windows (scripts/inkgt/mksources_u2ink.py: the 8 scans' onfly ink at level 0 = 2.4 um and pooled level 1 = 4.8 um, the
# validation rasters held out, the winding heads at 0), --init-from a u1 snapshot (the trunk, the winding / rv rows; the ink row fresh),
# u1's recipe (u1.train.sh, the latent flags of its relaunch line) with the rungs 2.4,4.8, STEPS (400) steps per arm, then the held-out
# score: scripts/inkgt/baseline.sh ckpt:<run>/last.ckpt:<ink channel> (3-D AUC / best F1 per crop vs ink-splat --face on the 16 Paris 4
# + 8 PHerc0139 / 1667 crops, predicted at 4.8 um; the teacher ink3dext: Paris 4 0.727, 0139 0.651, 1667 0.561, the CT floor 0.513).
# Arms (default: A B):
#   A  BCE over the known ink voxels + --ink-smooth 0.1 + --ink-dice 0.25 (the villa recipe without MIL)
#   B  A + --ink-mil 1 --ink-mil-r MILR (3: +-3 window voxels, 7 um at 2.4 / 14 um at 4.8; the trainer's default 10) on the pixels without
#      an open face (--ink-mil-mode 1)
# Pass: the ink AUC rises over the arm's first scored checkpoint and sits above the CT floor on every scan after STEPS; B >= A on the
# crops whose labels have many face-less pixels (Paris 4: 42%). Desk: GPU=1 (flock gpu1 bench lock, dev slice); pod: K set, LOCK= .
# Env: K (pod root; unset: the desk paths), U (binary with --ink-mil and the replay remap: master after milfix-20261112), INIT (u1 snapshot), SRCJ, STEPS (400),
#      P (896 pod; desk 384), GPU (0), MILR (3), MILMODE (1), SCORE (1: run baseline.sh after each arm), EVERY (200: checkpoint and score every EVERY steps).
set -uo pipefail; D=$(dirname "$(readlink -f "$0")"); ARMS=${*:-A B}
if [ -n "${K:-}" ]; then   # pod: what pod_pack.sh / the u2ink staging put under K
  R=${R:-$K/runs}; U=${U:-$K/ufsm/build/ufsm}; INIT=${INIT:-$K/ckpt/u1-resume.ckpt}; SRCJ=${SRCJ:-$K/inkgt/sources-u2ink.json}; BASIS=$K/small/klt1.lcb
  SC_ENV="D=$K/inkgt CROPS=$K/inkeval/crops CACHE=$K/cache"; P=${P:-896}; LOCK=${LOCK-}
else
  R=${R:-/vesuvius/ufsm/univ/u2ink/runs}; U=${U:-/vesuvius/ufsm/univ/u2ink/bin/ufsm}; INIT=${INIT:-/vesuvius/ufsm/v2/lat/runs/u1/snap/8000.ckpt}
  SRCJ=${SRCJ:-/vesuvius/ufsm/inkgt/sources-u2ink.json}; BASIS=/vesuvius/ufsm/v2/lat/codec/klt1.lcb; SC_ENV=""; P=${P:-384}; LOCK=${LOCK-/tmp/gpu1-bench.lock}
fi
STEPS=${STEPS:-400}; GPU=${GPU:-0}; MILR=${MILR:-3}; EVERY=${EVERY:-200}; mkdir -p $R
for f in $U $INIT $SRCJ $BASIS; do [ -e $f ] || { echo "missing $f"; exit 1; }; done
grep -aq -- "--ink-mil-r R > 0" $U || { echo "$U has no --ink-mil (build master after milfix-20261112)"; exit 1; }
HEADS="--task unwrap2 --rv 2 --rv-iso 64 --wind4-lambda 0.25 --density-scale 4 --density-code 1 --packed-weight 8 --normal-step 1 --ink-head 1"
LAT="--lat 4 --lat-basis $BASIS --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --replay 3 --um-rungs 2.4,4.8 --rung-shift-p 0.2 --film 2 --fstem 16 --fstem-drop 0.25"
INK="--ink-smooth 0.1 --ink-dice 0.25 --ink-bias -1"   # (the fresh ink row: logit -1, the GT's ~0.2-0.3 ink share)
for arm in $ARMS; do
  case $arm in A) X="$INK";; B) X="$INK --ink-mil 1 --ink-mil-r $MILR --ink-mil-mode ${MILMODE:-1}";; *) echo "unknown arm $arm"; exit 2;; esac
  n=u2ink$arm; o=$R/$n
  if [ ! -f $o/last.ckpt ] || [ "$(awk '/^step +[0-9]+ / {s=$2} END {print s + 0}' $o.log 2>/dev/null)" -lt $STEPS ]; then
    echo "== $(date '+%F %T') arm $arm: $o"
    (cd $R && R=$R U=$U SRCJ=$SRCJ GPU=$GPU LOCK=$LOCK P=$P ARCH="--widths 96,128,160,192" WORKERS=${WORKERS:-11} SLOTS=${SLOTS:-13} MEMMODE=auto \
       ALR=0.0001 MLR=0.002 WIND4L=0.25 ISO=64 MINMEM=${MINMEM:-16} SNAP=$EVERY CKPT=$EVERY WARM=50 LOGE=10 RELAUNCH=0 \
       HEADS="$HEADS" EXTRA="$LAT $X --init-from $INIT" bash $D/u1.train.sh $n $STEPS)
  fi
  ch=$(head -c 200000 $o/last.ckpt | head -1 | grep -o '"ink":[0-9]*' | head -1 | cut -d: -f2)
  [ -n "$ch" ] || { echo "arm $arm: no ink channel in $o/last.ckpt"; continue; }
  grep '^step' $o.log | awk '{for (i = 1; i <= NF; i++) if ($i == "ink" && $(i+1) == "bce") b = $(i+2); else if ($i == "ink" && $(i+1) == "mil") m = $(i+2); s = $2} NR % 10 == 0 {print "  step " s " ink bce " b (m != "" ? " mil " m : "")}' | tail -8
  [ "${SCORE:-1}" = 1 ] || continue
  for ck in $o/snap/*.ckpt $o/last.ckpt; do [ -f $ck ] || continue
    st=$(basename $ck .ckpt); env $SC_ENV U=$U O=$o/score/$st bash $D/../inkgt/baseline.sh ckpt:$ck:$ch $GPU | sed "s/^/  $arm $st: /"
    python3 -I - $o/score/$st/ckpt_$(basename $ck .ckpt)_ch$ch/vscores.tsv <<'PY'
import sys, collections
a = collections.defaultdict(list)
for l in open(sys.argv[1]):
    f = l.rstrip("\n").split("\t"); n = f[0]; s = "Paris4" if n.startswith("w") else "0139" if "0139" in n else "1667" if "1667" in n else n
    a[s].append(float(f[5]))
t = {"Paris4": 0.727, "0139": 0.651, "1667": 0.561}
print("  3-D AUC by scan: " + ", ".join(f"{s} {sum(v) / len(v):.3f} (teacher {t.get(s, float('nan')):.3f}, {len(v)} crops)" for s, v in sorted(a.items())))
PY
  done
done

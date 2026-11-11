#!/usr/bin/env bash
# scripts/univ/pod_univ.sh [STEP...]: make a fresh GPU pod ready for the universal-net runs (u1 resumed there, or U3), restartable step by
# step (each skips finished work). K (default /workspace/univ) holds what scripts/univ/pod_pack.sh staged from the desk (small/, gt/p0139-q,
# gt/1667/{codesA,codesB,phase}, md5/, ckpt/) plus everything built here. Steps (default: setup kit 1667 sources verify):
#   setup    limits, deps, clone + build master into K/ufsm (scripts/pod/setup.sh). If K/ckpt/u1-commit.txt exists, also that commit
#            as a worktree K/ufsm-u1 (the resumed run keeps the binary it was trained with).
#   kit      the public-data labels (scripts/pod/datakit.sh: segments, Paris 4 June / July phase, PHerc0500P2, PHerc0139).
#   1667     PHerc1667 phase A48 / B48 at 4.8 um from the staged codes (scripts/univ/b1_relabel.sh, the GPU pair rule).
#   sources  K/sources-u1-pod.json: the desk's sources-u1.json with the CT from the volcomp mirror (K/cache) and the pod's label paths.
#   verify   the md5 samples of the desk's label shards against the rebuilt ones (pod_pack.sh md5/; a mismatch is reported, not fatal:
#            DATAKIT.md lists the sets that rebuild byte-identical).
#   bench    planner dry runs (UFSM_MEM_DRY=1) of the u1 recipe at P 896 / 1152 / 1280 on both GPUs, then 150 real steps at P (BENCHP,
#            default 1152; init wz896, the full augmentation set): samples/s and the loader wait. Raise WORKERS / SLOTS if the wait is high.
#   resume   u1 on this pod: K/runs/u1/last.ckpt = the staged checkpoint, scripts/univ/u1.train.sh with the u1 recipe (the relaunch
#            line's flags), the run's own commit's binary; the uw / FiLM / stem / optimizer state and the step come from the checkpoint.
# Env: K, T (threads; default cgroup CPUs - 4), GPUS (0,1), SRCJ (sources, default K/sources-u1-pod.json), DRYP (bench dry-run P list), BENCHP, P (u1: 896), WORKERS / SLOTS (11 / 13 per the desk; raise with RAM).
set -uo pipefail; K=${K:-/workspace/univ}; D=$(dirname "$(readlink -f "$0")"); ST=${*:-setup kit 1667 sources verify}
GPUS=${GPUS:-0,1}; mkdir -p $K/runs $K/cache; cd $K
NC=$(python3 -c "import os; print(len(os.sched_getaffinity(0)))"); [ -r /sys/fs/cgroup/cpu.max ] && read -r q per < /sys/fs/cgroup/cpu.max && [ "$q" != max ] && NC=$((q / per)); T=${T:-$((NC > 8 ? NC - 4 : 4))}
U=$K/ufsm/build/ufsm; UR=$K/ufsm-u1/build/ufsm; [ -f $K/ckpt/u1-commit.txt ] || UR=$U
say() { echo "== $(date '+%F %T') pod_univ: $*"; }
# u1run NAME STEPS P BINARY: the u1 recipe (scripts/univ/u1.train.sh + the desk relaunch line's flags; auto-resume from K/runs/NAME/last.ckpt,
# else --init-from wz896), CT from the mirror. The bench / dry runs use the same launcher so their loader wait includes the augmentations.
u1run() { (cd $K/runs && R=$K/runs U=$4 SRCJ=${SRCJ:-$K/sources-u1-pod.json} GPU=$GPUS LOCK= P=$3 ARCH="--widths 96,128,160,192" WORKERS=${WORKERS:-11} \
  SLOTS=${SLOTS:-13} MEMMODE=auto ALR=0.0001 MLR=0.002 WIND4L=0.25 ISO=64 MINMEM=${MINMEM:-16} SNAP=1000 \
  EXTRA="--lat 4 --lat-basis $K/small/klt1.lcb --lat-c 29 --lat-step 0.25 --lat-step-jitter 0.3 --lat-guide 3 --replay 3 --um-rungs 4.8,9.6:2,19.2 --rung-shift-p 0.2 --film 2 --fstem 16 --fstem-drop 0.25 --uw 1 --grad-cos 500 --init-from $K/ckpt/wz896.ckpt" \
  nohup bash $D/u1.train.sh $1 $2 >> $K/runs/$1.nohup 2>&1 < /dev/null); }
for s in $ST; do say $s; case $s in
  setup) POD=$K BRANCH=${BRANCH:-master} bash $D/../pod/setup.sh || exit 1
    if [ -f $K/ckpt/u1-commit.txt ] && [ ! -x $UR ]; then c=$(cat $K/ckpt/u1-commit.txt)
      git -C $K/ufsm worktree add -f $K/ufsm-u1 $c && make -C $K/ufsm-u1 -j $T build/ufsm > $K/ufsm-u1.build.log 2>&1 || { tail -5 $K/ufsm-u1.build.log; exit 1; }
      say "u1 binary from $c: $UR"; fi;;
  kit) U=$U T=$T bash $D/../pod/datakit.sh $K segs p4 0500 0139 || exit 1;;
  1667) for S in 1667A 1667B; do K=$K G0=$K/gt/1667 AX=$K/small/axis/PHerc1667-unroll-axis.json U=$U GPU=${GPUS%%,*} T=$T bash $D/b1_relabel.sh $S || exit 1; done;;
  sources) python3 -I - $K <<'PY' || exit 1
import json, sys
K = sys.argv[1]; M = 'https://dl.ash2txt.org/community-uploads/forrest/volcomp'
S = json.load(open(f'{K}/small/sources-u1-desk.json'))
ax = {'/vesuvius/ufsm/teacher/axis-0.json': 'paris4-axis-0.json', '/mnt/scratch/pods/PHerc0139/unroll/axis.json': 'PHerc0139-unroll-axis.json',
      '/vesuvius/ufsm/v2/rv/gt/axis/PHerc0500P2-20250820143440.json': 'PHerc0500P2-20250820143440.json',
      '/vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json': 'PHerc1667-umbilicus.json'}
ph = {'/vesuvius/ufsm/v2/rv/gt/p4full/phaseA': 'gt/paris4/phaseA', '/vesuvius/ufsm/v2/rv/gt/p4full/phaseB': 'gt/paris4/phaseB',
      '/vesuvius/ufsm/v2/rv/gt/p0139full/phase': 'gt/p0139full/phase', '/vesuvius/ufsm/v2/rv/gt/p0500p2full/phase': 'gt/PHerc0500P2-wind/phase',
      '/vesuvius/ufsm/gt/PHerc1667-wind-20261016/phase': 'gt/1667/phase', '/mnt/scratch/univ/b1/gt/1667/phaseA48': 'gt/1667/phaseA48',
      '/mnt/scratch/univ/b1/gt/1667/phaseB48': 'gt/1667/phaseB48'}
for s in S['sources']:
    if s['root'] != M: s['root'] = M; a, b = s['ct'].split('/', 1); s['ct'] = f'{a}/volumes/{b}'   # (the desk's v2/ct = the mirror's <scroll>/volumes/<scan>)
    s['axis'] = f"{K}/small/axis/{ax[s['axis']]}"
    if 'holdout_file' in s: s['holdout_file'] = f'{K}/small/holdouts.json'
    p = s['targets']['band']['phase']; p['root'] = f"{K}/{ph[p['root']]}"
S['cache'] = f'{K}/cache'
json.dump(S, open(f'{K}/sources-u1-pod.json', 'w'), indent=1)
import os
for s in S['sources']:
    r = s['targets']['band']['phase']['root']; print(f"{s['name']}: {'ok' if os.path.exists(r + '/zarr.json') else 'MISSING ' + r}")
PY
    ;;
  verify) for f in $K/md5/*.md5; do n=$(basename $f .md5)
      case $n in p4A) d=gt/paris4/phaseA;; p4B) d=gt/paris4/phaseB;; p0139) d=gt/p0139full/phase;; p0500) d=gt/PHerc0500P2-wind/phase;; 1667A48) d=gt/1667/phaseA48;; 1667B48) d=gt/1667/phaseB48;; esac
      [ -d $K/$d ] || { echo "$n: $d not built"; continue; }
      echo "$n: $( (cd $K/$d && md5sum -c $f 2>/dev/null) | grep -c ': OK$') of $(wc -l < $f) sampled shards identical to the desk's"; done;;
  bench) for P in ${DRYP:-896 1152 1280}; do UFSM_MEM_DRY=1 UFSM_MEM_REPORT=1 RELAUNCH=0 u1run dry$P 10 $P $U; echo "P $P: $(grep -m1 'GB free per GPU\|no mode fits' $K/runs/dry$P.log | cut -c1-200)"; done
    BP=${BENCHP:-1152}; RELAUNCH=0 LOGE=25 u1run bench$BP 150 $BP $U
    grep '^step' $K/runs/bench$BP.log | tail -3 | cut -c1-220; grep -iE 'wait|samp/s' $K/runs/bench$BP.log | tail -2 | cut -c1-220;;
  resume) [ -f $K/ckpt/u1-resume.ckpt ] || { echo "no staged checkpoint (pod_pack.sh OUT SNAP)"; exit 1; }
    mkdir -p $K/runs/u1; [ -f $K/runs/u1/last.ckpt ] || cp $K/ckpt/u1-resume.ckpt $K/runs/u1/last.ckpt
    u1run u1 30000 ${P:-896} $UR &
    say "u1 launched (resumes $K/runs/u1/last.ckpt, $(cat $K/ckpt/u1-step.txt 2>/dev/null)); log $K/runs/u1.log, stop: touch $K/runs/u1.stop; kill -TERM \$(cat $K/runs/u1.pid)";;
  *) echo "unknown step $s"; exit 2;;
esac; done

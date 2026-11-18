#!/usr/bin/env bash
# scripts/inksmoke/podq.sh: the latent-ink smoke on a rented pod, unattended and restartable (2026-11-18; user: smoke test only).
# Default (FULL unset): a <= ~1 h check that the ink retrain trains: arm A only (human ink labels, u1-13000-w init) on the SCROLLS subset
# (PHerc1667, 0814, MAN5, 0009B: 166k of the 482k CT chunks; Paris 4 41 % and 0139 21 % of the prefetch are left out), its CT prefetched
# while narrow + weights run, STEPS 200 with a step-10 speed gate (MINSPS10 0.03 samples/s), the step-50 gate (MINSPS 0.1), a wall cap
# (ARMMAX 2700 s from the trainer's start) and a step-200 sanity (finite losses, ink bce of the last 3 logged lines vs the first 3); then
# the trainer stops and K/out holds what the desk needs (podq.log, steptimeA.txt, smokeA.txt, the run log, the step-200 ckpt, memory.events).
# FULL=1: the original ~10.5 h queue (all six scrolls, arm A 3000 steps, teach, arm B 3000 steps): only with a new user OK.
# Every stage leaves K/state/<stage>.done and is skipped once done; smoke and full keep separate runs / arms / state names. Stages:
#   setup   ufsm COMMIT cloned into K/ufsm, microc at the pin in ~/.local/mcc (a no-op when it is there: an install into another prefix
#           failed on pod B, stddef.h not found), build/ufsm + build-verify/test_unet
#           (-j BJ under /root/build.lock), mcudahw JIT cache warmed (CPU)
#   gt      the human ink GT from the public buckets (gtfetch.sh K/kit/gt.tsv: size + md5 checked) -> K/inkgt (the kit's segs-*.json,
#           sources-H0.json and axis/ already point there); in the background from the start
#   prefetch K/arms/TAG/sources-A.json (mkarms.py: weights capped, per-scroll holdout files) and its CT (ufsm prefetch, 48 threads;
#           pod B 11-18: all six scrolls 481,871 chunks, 47 GB at 30 MB/s in 42 min) in the background beside narrow + weights
#   narrow  test_unet~narrow (the RTX PRO fault class, fixed ab60f6ba) under GPU lock + timeout
#   weights reader-v3, ink9um seeds 42 / 43 from HuggingFace at pinned revisions, sha256 checked; the u1 trunk copied once (sha1 checked)
#   armA    trainarm.sh (GPU lock held for the arm); the gates above; K/steptimeA.txt at the first step line (step 10)
#   (FULL)  teach: reader3 + ink9um maps of every labelled mesh, ink-teach per scroll; armB: sources-B.json, 3000 steps
# Then K/state/ALLDONE: the pod is idle. Smoke time on a pod (estimate): setup ~5 min, prefetch ~15 min (166k chunks, beside narrow +
# weights), trainer start (source index, ink targets) ~10 min, 200 steps ~5-10 min: ~35-45 min, capped by ARMMAX.
# Env: K (/root/ink), COMMIT (required: the master commit), FULL (0), LOCK (/root/gpu.lock), BJ (6), GPU (0), STEPS (200 / 3000),
#      SCROLLS (the subset above / all), MINSPS10, MINSPS, ARMMAX (2700 / 0 = none), WORKERS / SLOTS (1 / 2: the desk sampler-only check
#      of arm A held 23-24 GB RSS at 2 / 2 before any pinned memory; the pod cgroup has 31 GB). Run it from the kit copy (K/kit/podq.sh:
#      it clones the repo itself), detached: COMMIT=... setsid nohup bash K/kit/podq.sh > K/podq.out 2>&1 &
set -uo pipefail
K=${K:-/root/ink}; COMMIT=${COMMIT:?COMMIT}; LOCK=${LOCK:-/root/gpu.lock}; BJ=${BJ:-6}; GPU=${GPU:-0}; FULL=${FULL:-0}
if [ "$FULL" = 1 ]; then TAG=full; export STEPS=${STEPS:-3000} SNAP=${SNAP:-1000} CKPT=${CKPT:-500} SCROLLS=${SCROLLS:-}; ARMMAX=${ARMMAX:-0}
else TAG=smoke; export STEPS=${STEPS:-200} SNAP=${SNAP:-200} CKPT=${CKPT:-100} SCROLLS=${SCROLLS-PHerc1667,PHerc0814,PHercMAN5,PHerc0009B}; ARMMAX=${ARMMAX:-2700}; fi
S=$K/state; L=$K/podq.log; O=$K/out; RD=$K/runs/$TAG; AD=$K/arms/$TAG; mkdir -p $S $RD $AD $O $K/teach $K/maps $K/weights $K/ckpt
say() { echo "$(date -u '+%F %T') podq: $*" | tee -a $L; echo "$*" > $K/status; }
die() { say "FAILED: $*"; touch $S/FAILED; exit 1; }
done_() { [ -e $S/$1.done ]; }; mark() { touch $S/$1.done; say "$1 done"; }
rm -f $S/FAILED $S/ALLDONE; say "start (commit $COMMIT, $TAG: STEPS $STEPS, scrolls ${SCROLLS:-all})"
T0=$(date +%s); el() { echo "$(( ($(date +%s) - T0) / 60 )) min"; }
U=$K/ufsm/build/ufsm; D=$K/ufsm/scripts/inksmoke; export MCC_HOME=$HOME/.local/mcc
# --- gt: in the background from the start (network + CPU only) ---
if ! done_ gt; then ( bash $K/kit/gtfetch.sh $K/kit/gt.tsv $K/inkgt > $K/gt.log 2>&1 && touch $S/gt.done ) & GTPID=$!; else GTPID=; fi
# --- setup ---
if ! done_ setup; then
  say "setup: clone + build $COMMIT"
  [ -d $K/ufsm/.git ] || git clone -q https://github.com/SuperOptimizer/ufsm $K/ufsm || die "clone"
  git -C $K/ufsm fetch -q origin && git -C $K/ufsm checkout -q --detach $COMMIT || die "checkout $COMMIT"
  bash $K/ufsm/scripts/toolchain/install_mcc.sh > $K/mcc.log 2>&1 || die "install_mcc (log $K/mcc.log)"
  ( cd $K/ufsm && flock /root/build.lock bash -c "cmake --preset release && nice -n 5 cmake --build build -j $BJ --target ufsm && cmake --preset verify && nice -n 5 cmake --build build-verify -j $BJ --target test_unet" ) > $K/build.log 2>&1 \
    || die "build (log $K/build.log)"
  CUDA_VISIBLE_DEVICES=$GPU JITLOG=$K/jit.log bash $K/ufsm/scripts/toolchain/jit_warm.sh $U $BJ >> $L 2>&1
  CUDA_VISIBLE_DEVICES=$GPU JITLOG=$K/jit.log bash $K/ufsm/scripts/toolchain/jit_warm.sh $K/ufsm/build-verify/test_unet $BJ >> $L 2>&1
  mark setup
fi
[ -n "${GTPID:-}" ] && { wait $GTPID; GTPID=; }; done_ gt || die "gt fetch (log $K/gt.log)"
# --- prefetch (CPU + network, in the background beside narrow + weights) ---
mkarms() { [ -s $AD/sources-$1.json ] && return 0
  python3 -I $D/mkarms.py $K/inkgt/sources-H0.json $K/teach $K/kit/holdouts-20261118.json $AD $1 > $AD/mkarms$1.log 2>&1 || die "mkarms $1 (log $AD/mkarms$1.log)"; }
prefetch() {   # prefetch ARM: cold https CT read by one sampler worker gave 30 steps in 4 h on pod B (0.003 samples/s)
  done_ prefetch$1$TAG && return 0
  $U prefetch $AD/sources-$1.json --levels 1,2 --threads ${PFTH:-48} --P ${P:-1024} > $K/prefetch$1$TAG.log 2>&1 && touch $S/prefetch$1$TAG.done; }
mkarms A; PFPID=; if ! done_ prefetchA$TAG; then say "prefetch A ($TAG, scrolls ${SCROLLS:-all}) in the background"; prefetch A & PFPID=$!; fi
# --- narrow ---
if ! done_ narrow; then
  say "narrow: test_unet~narrow"
  ( cd $K/ufsm && flock -w 14400 $LOCK timeout -k 30 1800 env CUDA_VISIBLE_DEVICES=$GPU UFSM_P=32 UFSM_ACT_MX8=1 UFSM_GRAD_MX8=1 UFSM_LEAN=2 UFSM_CHUNK_UP=2 \
      UFSM_BIG_WIDTHS=32,64,96,64 UFSM_GRAD_TOL=0.5 ./build-verify/test_unet ) > $K/narrow.log 2>&1 || die "test_unet~narrow (log $K/narrow.log)"
  mark narrow
fi
# --- weights + init ---
if ! done_ weights; then
  get() { local url=$1 out=$2 sha=$3 cmd=${4:-sha256sum}; [ "$($cmd $out 2>/dev/null | cut -d' ' -f1)" = $sha ] && return 0
    curl -sfL --retry 5 --max-time 3600 -o $out.part "$url" && mv $out.part $out && [ "$($cmd $out | cut -d' ' -f1)" = $sha ] || die "fetch $out"; }
  get https://huggingface.co/domenicor046/reader-v3/resolve/aee1d64bced4eb35546348f3d276b9e2a0b99c46/reader-v3.pth $K/weights/reader-v3.pth \
      ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a
  get https://huggingface.co/scrollprize/ink_9um/resolve/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/hybrid_3d2d-seed42/step-075000.pth $K/weights/ink9um-s42.pth \
      e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab
  get https://huggingface.co/scrollprize/ink_9um/resolve/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/hybrid_3d2d-seed43/step-075000.pth $K/weights/ink9um-s43.pth \
      2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f
  [ "$(sha1sum $K/ckpt/u1-13000-w.ckpt 2>/dev/null | cut -d' ' -f1)" = 0d05642296829a4ca588bd7d91eb30f1470a3941 ] || cp /root/pod/ckpt/u1-13000-w.ckpt $K/ckpt/u1-13000-w.ckpt
  [ "$(sha1sum $K/ckpt/u1-13000-w.ckpt | cut -d' ' -f1)" = 0d05642296829a4ca588bd7d91eb30f1470a3941 ] || die "u1-13000-w sha1"
  mark weights
fi
if [ -n "$PFPID" ]; then wait $PFPID; fi; done_ prefetchA$TAG || die "prefetch A (log $K/prefetchA$TAG.log)"; say "prefetch A done ($(el); $(tail -1 $K/prefetchA$TAG.log | cut -c1-160))"
sps_of() {   # the samples/s of each logged step line: the fresh rate when --replay echoes windows (the echo affix), else the work rate
  grep -E '^step +[0-9]+ ' $1 | awk '{ if (match($0, /fresh [0-9.]+ samp\/s/)) { split(substr($0, RSTART, RLENGTH), f, " "); print f[2] }
    else { n = split($0, w, " "); for (i = n; i > 1; i--) if (w[i] == "samp/s") { print w[i - 1]; break } } }'; }
sanity() {   # sanity ARM: finite losses, the ink bce trend, the rates; K/out/smokeARM.txt; non-finite fails
  local a=$1 lg=$RD/inksmoke$1.log f=$O/smoke$1.txt
  { echo "arm $a ($TAG, scrolls ${SCROLLS:-all}): $(date -u '+%F %T'), $(el) since the queue start"
    echo "steps logged: $(grep -cE '^step +[0-9]+ ' $lg), last: $(grep -E '^step +[0-9]+ ' $lg | tail -1 | cut -c1-300)"
    echo "samples/s (fresh) per logged line: $(sps_of $lg | tr '\n' ' ')"
    grep -E '^step +[0-9]+ ' $lg | grep -oE 'ink bce [0-9.naif-]+' | awk '{v[NR] = $3} END {if (NR < 6) {print "ink bce: only " NR " logged lines"; exit}
      a = (v[1] + v[2] + v[3]) / 3; b = (v[NR] + v[NR - 1] + v[NR - 2]) / 3; printf "ink bce: first 3 lines %.4f, last 3 %.4f (%s)\n", a, b, b < a ? "falling: OK" : "NOT falling: WARN"}'
    echo "non-finite lines: $(grep -ciE 'non-finite|^step .*( nan| inf)' $lg)"; echo "trainer host RSS peak (30 s samples): ${RSSPK:-?} GB"
    for m in memory.events memory.peak; do [ -r /sys/fs/cgroup/$m ] && echo "$m: $(tr '\n' ' ' < /sys/fs/cgroup/$m)"; done
  } > $f; cat $f >> $L
  cp $lg $RD/inksmoke$a.resume.log $K/steptime$a.txt $O/ 2>/dev/null; ls $RD/inksmoke$a/snap/*.ckpt > /dev/null 2>&1 && cp $(ls -t $RD/inksmoke$a/snap/*.ckpt | head -1) $O/inksmoke$a-$TAG.ckpt
  cp $L $O/podq.log; [ "$(grep -ciE 'non-finite|^step .*( nan| inf)' $lg)" = 0 ] || die "arm $a: non-finite losses ($f)"; }
arm() {   # arm A | B
  local a=$1; done_ arm$a$TAG && return 0
  mkarms $a; [ $a = A ] || { prefetch $a; done_ prefetch$a$TAG || die "prefetch $a (log $K/prefetch$a$TAG.log)"; }
  say "arm $a: training $STEPS steps ($TAG, $(el))"; local t0=$(date +%s)
  ( R=$RD U=$U INIT=$K/ckpt/u1-13000-w.ckpt LATB=$K/kit/klt1.lcb GPU=$GPU LOCK=$LOCK RELAUNCH=3 WORKERS=${WORKERS:-1} SLOTS=${SLOTS:-2} bash $D/trainarm.sh $a $AD/sources-$a.json ) > $RD/arm$a.out 2>&1 &
  local tp=$! st=0 sps rss cg MEMS; RSSPK=0
  stoparm() { touch $RD/inksmoke$a.stop; kill $(cat $RD/inksmoke$a.pid 2>/dev/null) 2>/dev/null; sleep 30; sanity $a; die "$*"; }
  while kill -0 $tp 2>/dev/null; do
    st=$(awk '/^step +[0-9]+ / {s = $2} END {print s + 0}' $RD/inksmoke$a.log 2>/dev/null)
    rss=$(awk '/^(RssAnon|RssShmem)/ {s += $2} END {printf "%.1f", s / 1048576}' /proc/$(cat $RD/inksmoke$a.pid 2>/dev/null)/status 2>/dev/null)
    awk -v a=${rss:-0} -v b=${RSSPK:-0} 'BEGIN {exit !(a > b)}' && RSSPK=$rss   # the trainer's host RSS peak (30 s samples; the pod has 31 GB)
    cg=$(awk '{printf "%.1f", $1 / 1073741824}' /sys/fs/cgroup/memory.current 2>/dev/null); MEMS="RSS ${RSSPK:-?} GB peak, cgroup ${cg:-?} GB"
    if [ "${st:-0}" -ge 10 ] && [ ! -e $K/gate10$a$TAG.ok ]; then   # the first step line: report it, and stop a hopeless arm at once
      { grep -E '^step +[0-9]+ ' $RD/inksmoke$a.log | head -1; grep -E 'memory:|planner|batch' $RD/inksmoke$a.log | tail -3; } > $K/steptime$a.txt
      sps=$(sps_of $RD/inksmoke$a.log | head -1); say "arm $a: step $st at $(el) (trainer start + $(( ($(date +%s) - t0) / 60 )) min), $sps samples/s (gate ${MINSPS10:-0.03}), $MEMS"
      touch $K/gate10$a$TAG.ok; awk -v a=${sps:-0} -v b=${MINSPS10:-0.03} 'BEGIN {exit !(a < b)}' && stoparm "arm $a below the step-10 gate: $sps samples/s"; fi
    if [ "${st:-0}" -ge 50 ] && [ ! -e $K/gate$a$TAG.ok ]; then   # the step-50 gate: the mean of the logged lines
      sps=$(sps_of $RD/inksmoke$a.log | awk '{s += $1; n++} END {printf "%.3f", n ? s / n : 0}')
      grep -E '^step +[0-9]+ ' $RD/inksmoke$a.log | head -6 > $K/steptime$a.txt.new && mv $K/steptime$a.txt.new $K/steptime$a.txt
      echo "mean samples/s to step $st: $sps" >> $K/steptime$a.txt; say "arm $a: step $st at $(el), mean $sps samples/s (gate ${MINSPS:-0.1}), $MEMS"
      touch $K/gate$a$TAG.ok; awk -v a=$sps -v b=${MINSPS:-0.1} 'BEGIN {exit !(a < b)}' && stoparm "arm $a below the step-50 gate: $sps samples/s at step $st (fallback: --um-rungs 9.6 only)"; fi
    [ "$ARMMAX" -gt 0 ] && [ $(( $(date +%s) - t0 )) -gt "$ARMMAX" ] && stoparm "arm $a over ARMMAX ${ARMMAX} s at step $st"
    sleep 30; done
  wait $tp; st=$(awk '/^step +[0-9]+ / {s = $2} END {print s + 0}' $RD/inksmoke$a.log 2>/dev/null)
  sanity $a; [ "${st:-0}" -ge $STEPS ] || die "arm $a stopped at step $st (logs $RD/inksmoke$a.log, .resume.log)"
  mark arm$a$TAG
}
arm A
if [ "$FULL" != 1 ]; then cp $L $O/podq.log; touch $S/ALLDONE; say "ALLDONE: smoke arm A reached step $STEPS ($(el)); $O/smokeA.txt; the pod is idle"; cp $L $O/podq.log; exit 0; fi
if ! done_ teach; then
  for sc in PHercParis4 PHerc1667 PHerc0139 PHerc0814 PHerc0009B PHercMAN5; do
    U=$U GPU=$GPU LOCK=$LOCK bash $D/teach.sh $K $sc >> $L 2>&1 || die "teach $sc"
  done; mark teach
fi
arm B
touch $S/ALLDONE; say "ALLDONE: both arms trained; the pod is idle"

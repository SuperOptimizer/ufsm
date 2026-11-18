#!/usr/bin/env bash
# scripts/inksmoke/podq.sh: the latent-ink smoke on a rented pod, unattended and restartable (2026-11-18; user: smoke test only, ~10.5 h).
# Runs from K (/root/ink); every stage leaves K/state/<stage>.done and is skipped once done. Stages:
#   setup   ufsm COMMIT cloned into K/ufsm, microc at the pin in ~/.local/mcc (a no-op when it is there: an install into another prefix
#           failed on pod B, stddef.h not found), build/ufsm + build-verify/test_unet
#           (-j BJ under /root/build.lock), mcudahw JIT cache warmed (CPU)
#   narrow  test_unet~narrow (the RTX PRO fault class, fixed ab60f6ba) under GPU lock + timeout
#   gt      the human ink GT from the public buckets (gtfetch.sh K/kit/gt.tsv: size + md5 checked) -> K/inkgt (the kit's segs-*.json,
#           sources-H0.json and axis/ already point there)
#   weights reader-v3, ink9um seeds 42 / 43 from HuggingFace at pinned revisions, sha256 checked; the u1 trunk copied once (sha1 checked)
#   armA    K/arms/sources-A.json (mkarms.py A: weights capped, the per-scroll holdout files from the kit), its CT prefetched (ufsm prefetch,
#           48 threads), then arm A 3000 steps (trainarm.sh; GPU lock held for the arm); speed gate at step 50 (MINSPS 0.1 samples/s:
#           below it the arm is stopped and the queue fails); the first steps -> K/steptimeA.txt (report them at step 10)
#   teach   reader3 + ink9um maps of every labelled mesh, ink-teach per scroll (teach.sh; GPU lock per ink-ext call)
#   armB    K/arms/sources-B.json (mkarms.py B) and arm B 3000 steps
# Then K/state/ALLDONE: the pod is idle (the snapshots K/runs/inksmoke{A,B}/snap/{1000,2000,3000}.ckpt are scored on the desk).
# Env: K (/root/ink), COMMIT (required: the master commit), LOCK (/root/gpu.lock), BJ (6), GPU (0), STEPS (3000), WORKERS / SLOTS (1 / 2:
#      the desk sampler-only check of arm A held 23-24 GB RSS at 2 / 2 before any pinned memory; the pod cgroup has 31 GB). Run it from the
#      kit copy (K/kit/podq.sh: it clones the repo itself), detached: COMMIT=... setsid nohup bash K/kit/podq.sh > K/podq.out 2>&1 &
set -uo pipefail
K=${K:-/root/ink}; COMMIT=${COMMIT:?COMMIT}; LOCK=${LOCK:-/root/gpu.lock}; BJ=${BJ:-6}; GPU=${GPU:-0}; export STEPS=${STEPS:-3000}
S=$K/state; L=$K/podq.log; mkdir -p $S $K/runs $K/arms $K/teach $K/maps $K/weights $K/ckpt
say() { echo "$(date -u '+%F %T') podq: $*" | tee -a $L; echo "$*" > $K/status; }
die() { say "FAILED: $*"; touch $S/FAILED; exit 1; }
done_() { [ -e $S/$1.done ]; }; mark() { touch $S/$1.done; say "$1 done"; }
rm -f $S/FAILED; say "start (commit $COMMIT)"
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
[ -n "${GTPID:-}" ] && { wait $GTPID; }; done_ gt || die "gt fetch (log $K/gt.log)"
arm() {   # arm A | B
  local a=$1; done_ arm$a && return 0
  python3 -I $D/mkarms.py $K/inkgt/sources-H0.json $K/teach $K/kit/holdouts-20261118.json $K/arms $a > $K/arms/mkarms$a.log 2>&1 || die "mkarms $a (log $K/arms/mkarms$a.log)"
  # the arm's CT local first (pod B 11-18: cold https CT read by one sampler worker gave 30 steps in 4 h, 0.003 samples/s; ufsm prefetch
  # of every labelled window's CT, levels 1-2 (+ 4-5 for the draw cells), 48 threads: 481,871 chunks, 47 GB at 30 MB/s in 42 min)
  if ! done_ prefetch$a; then say "arm $a: prefetch (CT of every labelled window)"
    $U prefetch $K/arms/sources-$a.json --levels 1,2 --threads ${PFTH:-48} --P ${P:-1024} > $K/prefetch$a.log 2>&1 || die "prefetch $a (log $K/prefetch$a.log)"; mark prefetch$a; fi
  say "arm $a: training $STEPS steps"
  ( R=$K/runs U=$U INIT=$K/ckpt/u1-13000-w.ckpt LATB=$K/kit/klt1.lcb GPU=$GPU LOCK=$LOCK RELAUNCH=3 WORKERS=${WORKERS:-1} SLOTS=${SLOTS:-2} bash $D/trainarm.sh $a $K/arms/sources-$a.json ) > $K/runs/arm$a.out 2>&1 &
  local tp=$! st=0
  while kill -0 $tp 2>/dev/null; do
    st=$(awk '/^step +[0-9]+ / {s = $2} END {print s + 0}' $K/runs/inksmoke$a.log 2>/dev/null)
    if [ ! -s $K/steptime$a.txt ] && [ "${st:-0}" -ge 60 ]; then   # (the first 50 steps after the warm-up line: s / step from the log's wall stamps)
      grep -E '^step +[0-9]+ ' $K/runs/inksmoke$a.log | head -8 > $K/steptime$a.txt; grep -E 's/step|samp/s|Mvox/s' $K/runs/inksmoke$a.log | tail -3 >> $K/steptime$a.txt
      say "arm $a: step $st reached ($(date -u +%T)); first steps in $K/steptime$a.txt"
    fi
    if [ "${st:-0}" -ge 50 ] && [ ! -e $K/gate$a.ok ]; then   # the speed gate at step 50: the mean samples/s of the logged 10-step lines
      sps=$(grep -E '^step +[0-9]+ ' $K/runs/inksmoke$a.log | grep -oE '[0-9.]+ samp/s' | awk '{s += $1; n++} END {printf "%.3f", n ? s / n : 0}')
      say "arm $a: step $st, $sps samples/s (gate ${MINSPS:-0.1})"
      if awk -v a=$sps -v b=${MINSPS:-0.1} 'BEGIN {exit !(a < b)}'; then touch $K/runs/inksmoke$a.stop; kill $(cat $K/runs/inksmoke$a.pid) 2>/dev/null
        die "arm $a below the speed gate: $sps samples/s at step $st (fallback: --um-rungs 9.6 only, Paris 4 dropped)"; fi
      touch $K/gate$a.ok; fi
    sleep 60; done
  wait $tp; st=$(awk '/^step +[0-9]+ / {s = $2} END {print s + 0}' $K/runs/inksmoke$a.log 2>/dev/null)
  [ "${st:-0}" -ge $STEPS ] || die "arm $a stopped at step $st (logs $K/runs/inksmoke$a.log, .resume.log)"
  ls $K/runs/inksmoke$a/snap/*.ckpt >> $L 2>&1; mark arm$a
}
arm A
if ! done_ teach; then
  for sc in PHercParis4 PHerc1667 PHerc0139 PHerc0814 PHerc0009B PHercMAN5; do
    U=$U GPU=$GPU LOCK=$LOCK bash $D/teach.sh $K $sc >> $L 2>&1 || die "teach $sc"
  done; mark teach
fi
arm B
touch $S/ALLDONE; say "ALLDONE: both arms trained; the pod is idle"

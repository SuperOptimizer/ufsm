#!/usr/bin/env bash
# scripts/pod/podq2.sh: the next pod session's queue (2026-11-18, lead): (1) a production-style full PHerc1667 unroll at master with every
# speedup (INKMODE=r3 + R3ONCE ink-once, the m7 prefetch beside the fields, render's CT alias + --volume-out 0, the fields predict's fused
# head slab / MX head / fstem kernels: all master defaults), its results pushed to the desk by this job AS EACH STAGE GROUP LANDS (podun
# 11-17: everything not pulled before the pod was shut down was lost), GT-scored there (scripts/pod/score1667.sh, on the desk); (2) then
# inksdk's latent-ink smoke (the kit's podq.sh, /vesuvius/ufsm/inksmoke/kit: CT prefetch first, its step-10 / step-50 gates).
# Unattended and restartable: every step leaves Q/state/<step>.done and is skipped once done; re-run the same command after any failure.
#   setup   scripts/pod/setup.sh PHerc1667 (clone COMMIT, mcc build, smoke tests, CT levels from the mirror, reader-v3) + the fields CK
#           (CKURL fetched, or CK already on the pod; sha1 checked when CKSHA1 is given)
#   desk    the desk is reachable for the pushes (DESK=user@host, DESKPORT, DESKKEY; a test file round trip): else the queue stops here
#           with the podpull.sh command for the desk instead (unrolls must never finish unsynced)
#   unroll  scripts/pod/unroll.sh PHerc1667 (RECTOSP 1, INKMODE r3, R3ONCE 1, PERWRAP 1, KEEPINK 1 (the per-wrap ink maps for the
#           ink GT score), CLEAN 0: the GT score reads the fields), with a
#           background pusher: progress.log / state / logs every 2 min, each stage group's outputs once its .done marker exists
#   push    the final sync (fields/rsp, ref-cf, joined, wrapcheck.tsv, qa, img, prev, strip/img_fill + img_pap + distort, up, ink/wraps + the stripes' ink/r3/strip maps)
#           and the stage table (stagetable.tsv: stage, start, end, wall s, CPU s, max RSS GB from progress.log) -> DESKDIR
#   smoke   the ink smoke: K=/root/ink, the kit copied from the desk beforehand (tiny files; KIT), COMMIT=COMMIT bash $K/kit/podq.sh
#           (its own state; its gates stop it) -- in the foreground of this queue, so STOPQ / a failure stops here
# Env: POD (/root/pod), Q ($POD/q2), COMMIT (required: the master commit to run), CK (fields checkpoint path on the pod; default
#   $POD/ckpt/u1-13790.ckpt, the production fields model (lead 11-18): the desk's /vesuvius/ufsm/v2/lat/runs/u1-donor-13790.ckpt, 201 MB,
#   not on the mirror: scp it beforehand, ~70 s at the desk's uplink) or CKURL, CKSHA1 (default: u1 13790's 8854e4e5...), DESK (forrest@<desk host reachable from the pod>), DESKPORT (22), DESKKEY (~/.ssh/id_ed25519), DESKDIR
#   (/vesuvius/ufsm/podun/PHerc1667-q2; DESK=none: a pod that cannot reach the desk: no pushes, the desk pulls with
#   scripts/pod/q2pull.sh, which ends on RU/.final (unroll) / RU/.alldone (queue)), TH / PTH / ITH (threads; default the cgroup's), K (/root/ink), NOSMOKE=1 (stop after the unroll), MINDISK (120 GB free on / before the unroll).
# Run detached on the pod: COMMIT=... DESK=forrest@HOST setsid nohup bash $POD/ufsm/scripts/pod/podq2.sh > $POD/podq2.out 2>&1 < /dev/null &
# (from a fresh pod: copy this file there first, or clone the repo; setup clones COMMIT into $POD/ufsm and the queue then runs the same
# script from the repo copy). Stop after the current step: touch $Q/STOPQ.
set -uo pipefail
POD=${POD:-/root/pod}; Q=${Q:-$POD/q2}; COMMIT=${COMMIT:?COMMIT=<master commit>}; S=PHerc1667
DESK=${DESK:-}; DESKPORT=${DESKPORT:-22}; DESKKEY=${DESKKEY:-$HOME/.ssh/id_ed25519}; DESKDIR=${DESKDIR:-/vesuvius/ufsm/podun/PHerc1667-q2}
K=${K:-/root/ink}; MINDISK=${MINDISK:-120}; ST=$Q/state; L=$Q/podq2.log; mkdir -p $ST
say() { echo "$(date -u '+%F %T') podq2: $*" | tee -a $L; echo "$*" > $Q/status; }
die() { say "FAILED: $*"; touch $ST/FAILED; exit 1; }
done_() { [ -e $ST/$1.done ]; }; mark() { touch $ST/$1.done; say "$1 done"; }
stopq() { [ -e $Q/STOPQ ] && { say "STOPQ: stopping before $1"; exit 0; }; return 0; }
SSH="ssh -p $DESKPORT -i $DESKKEY -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
push() { rsync -a --partial -e "$SSH" "$@" 2>> $Q/push.err; }   # push SRC.. DESK:DST (rsync, resumable)
rm -f $ST/FAILED; say "start (commit $COMMIT)"

# --- setup: the repo at COMMIT, the build, the CT, the fields checkpoint ---
stopq setup
if ! done_ setup; then
  SU=${SETUP:-$POD/ufsm/scripts/pod/setup.sh}; [ -s $SU ] || { SU=$(dirname "$(readlink -f "$0")")/setup.sh; [ -s $SU ] || { SU=$Q/setup.sh;
    curl -sf -o $SU https://raw.githubusercontent.com/SuperOptimizer/ufsm/$COMMIT/scripts/pod/setup.sh || die "setup.sh at $COMMIT"; }; }   # (a fresh pod: no clone yet)
  BRANCH=$COMMIT POD=$POD bash $SU $S > $Q/setup.out 2>&1 || die "setup.sh (see $Q/setup.out)"
  ( cd $POD/ufsm && git rev-parse HEAD | grep -q "^$COMMIT" ) || die "the repo is not at $COMMIT"
  CK=${CK:-$POD/ckpt/u1-13790.ckpt}; [ "$CK" = $POD/ckpt/u1-13790.ckpt ] && CKSHA1=${CKSHA1:-8854e4e5bd8ede2133c70aada6032a5b07715bb0}
  if [ ! -s $CK ]; then [ -n "${CKURL:-}" ] || die "no fields checkpoint: CK=$CK missing and no CKURL"; curl -fL --retry 5 -C - -o $CK.part "$CKURL" && mv $CK.part $CK || die "fetch $CKURL"; fi
  [ -z "${CKSHA1:-}" ] || [ "$(sha1sum $CK | cut -d' ' -f1)" = "$CKSHA1" ] || die "checkpoint $CK: sha1 mismatch"
  echo "$CK" > $Q/ck.path; mark setup
fi
CK=$(cat $Q/ck.path); U=$POD/ufsm/build/ufsm; RU=$POD/unroll/$S

# --- desk: the pushes must work before any result exists ---
stopq desk
if ! done_ desk; then
  [ "$DESK" = none ] && { say "DESK=none: no pushes; the desk pulls (scripts/pod/q2pull.sh)"; mark desk; }
fi
if ! done_ desk; then
  [ -n "$DESK" ] || die "DESK unset: set DESK=user@host (reachable from the pod), or run on the desk: PODH=root@<pod> bash scripts/pod/podpull.sh"
  echo "podq2 $(date -u +%s) $COMMIT" > $Q/desk.probe
  $SSH $DESK "mkdir -p $DESKDIR" && push $Q/desk.probe $DESK:$DESKDIR/desk.probe && $SSH $DESK "cat $DESKDIR/desk.probe" | grep -q "$COMMIT" \
    || die "the desk ($DESK:$DESKPORT) is not reachable for pushes ($Q/push.err); pull from the desk instead: PODH=root@<pod> bash scripts/pod/podpull.sh"
  mark desk
fi

# --- unroll (+ the background pusher) ---
stopq unroll
pusher() {   # every 2 min: the small rows; each stage group's outputs once its marker exists (once each)
  while :; do
    [ -d $RU ] && { push $RU/progress.log $RU/state $RU/logs $DESK:$DESKDIR/ ; for f in box.txt axis.json wrapcheck.tsv trim.pap slabs.txt; do [ -e $RU/$f ] && push $RU/$f $DESK:$DESKDIR/; done; }
    [ -e $RU/state/ref.done ] && [ ! -e $Q/pushed.ref ] && push $RU/ref-cf $DESK:$DESKDIR/ && touch $Q/pushed.ref
    [ -e $RU/state/join.done ] && [ ! -e $Q/pushed.join ] && push $RU/joined $DESK:$DESKDIR/ && touch $Q/pushed.join
    [ -e $RU/state/snap.done ] && [ ! -e $Q/pushed.wraps ] && push $RU/wraps_snap $RU/qa $DESK:$DESKDIR/ && touch $Q/pushed.wraps
    [ -e $RU/state/stripimg.done ] && [ ! -e $Q/pushed.strip ] && { $SSH $DESK "mkdir -p $DESKDIR/strip" && push $RU/strip/img_fill $RU/strip/img_pap $RU/strip/distort $DESK:$DESKDIR/strip/ && touch $Q/pushed.strip; }
    [ -e $RU/state/perwrap.done ] && [ ! -e $Q/pushed.img ] && { $SSH $DESK "mkdir -p $DESKDIR/ink" && push $RU/img $RU/prev $RU/up $DESK:$DESKDIR/ && push $RU/ink/wraps $DESK:$DESKDIR/ink/ && touch $Q/pushed.img; }
    [ -e $Q/pusher.stop ] && return 0; sleep 120
  done
}
if ! done_ unroll; then
  w=0; while [ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -lt $MINDISK ]; do [ $w = 0 ] && say "disk: < $MINDISK GB free on /: waiting"; w=1; sleep 60; done
  rm -f $Q/pusher.stop; PP=; [ "$DESK" != none ] && { pusher & PP=$!; }
  say "unroll start (CK $CK)"
  ( cd $POD/ufsm && export TH=${TH:-} PTH=${PTH:-} ITH=${ITH:-}; [ -z "$TH" ] && unset TH PTH ITH; CK=$CK U=$U RECTOSP=1 INKMODE=r3 R3ONCE=1 PERWRAP=1 KEEPINK=1 CLEAN=0 timeout -k 60 ${QTIMEOUT:-21600} bash scripts/pod/unroll.sh $S > $Q/unroll.out 2>&1 ); rc=$?
  touch $Q/pusher.stop; [ -n "$PP" ] && while kill -0 $PP 2> /dev/null; do sleep 5; done
  [ $rc = 0 ] && [ -e $RU/state/stripimg.done ] && [ -e $RU/state/perwrap.done ] || die "unroll rc $rc (no stripimg.done / perwrap.done: $Q/unroll.out, $RU/progress.log)"
  mark unroll
fi

# --- push: the final sync + the stage table ---
stopq push
if ! done_ push; then
  awk '/ start$/ {s[$3] = $1 " " $2} / done: / {st = $3; sub(/:$/, "", st); w = $5; c = $8; r = $11; printf "%s\t%s\t%s %s\t%s\t%s\t%s\n", st, s[st], $1, $2, w, c, r}' \
    $RU/progress.log | awk -F'\t' 'BEGIN {print "stage\tstart\tend\twall_s\tcpu_s\tmax_rss_gb"} {print}' > $Q/stagetable.tsv
  cp -f $Q/stagetable.tsv $RU/stagetable.tsv
  if [ "$DESK" = none ]; then touch $RU/.final; mark push; say "unroll final ($RU/.final): the desk pulls it (scripts/pod/q2pull.sh)"; fi
fi
if ! done_ push; then
  $SSH $DESK "mkdir -p $DESKDIR/fields $DESKDIR/ink $DESKDIR/strip" || die "desk mkdir"
  push $RU/fields/rsp $DESK:$DESKDIR/fields/ && push $RU/ref-cf $RU/joined $RU/wrapcheck.tsv $RU/qa $RU/img $RU/prev $RU/up $RU/progress.log $RU/state $RU/logs $Q/stagetable.tsv $DESK:$DESKDIR/ \
    && push $RU/strip/img_fill $RU/strip/img_pap $RU/strip/distort $DESK:$DESKDIR/strip/ && push $RU/ink/wraps $RU/ink/r3/strip $DESK:$DESKDIR/ink/ || die "final push ($Q/push.err)"
  $SSH $DESK "touch $DESKDIR/.synced" || die "desk marker"
  mark push; say "synced to $DESK:$DESKDIR; on the desk: bash scripts/pod/score1667.sh $DESKDIR"
fi

# --- smoke: inksdk's latent-ink smoke (the kit's own queue, gates and state) ---
[ "${NOSMOKE:-0}" = 1 ] && { say "NOSMOKE: done after the unroll"; exit 0; }
stopq smoke
if ! done_ smoke; then
  [ -s $K/kit/podq.sh ] || die "no ink smoke kit at $K/kit (copy /vesuvius/ufsm/inksmoke/kit from the desk: small files only)"
  ( cd $POD/ufsm && git merge-base --is-ancestor 44357ead $COMMIT ) || die "COMMIT predates 44357ead: the kit's ~40 min smoke (mkarms SCROLLS) needs it (inksdk 11-18)"
  say "smoke start (K $K)"
  COMMIT=$COMMIT K=$K bash $K/kit/podq.sh > $K/podq.out 2>&1; rc=$?
  [ $rc = 0 ] && [ -e $K/state/ALLDONE ] || die "ink smoke rc $rc ($K/podq.log)"
  [ "$DESK" != none ] && $SSH $DESK "mkdir -p $DESKDIR/smoke" && push $K/podq.log $K/steptimeA.txt $K/out $DESK:$DESKDIR/smoke/ 2> /dev/null   # (out/: smokeA.txt, run / resume logs, the step-200 ckpt ~80 MB)
  mark smoke
fi
say "ALL DONE: the pod is idle"; touch $ST/ALLDONE $RU/.alldone

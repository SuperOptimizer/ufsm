#!/bin/bash
# scripts/pod/q2pull.sh (desk): pull a podq2.sh run (DESK=none: the pod cannot reach the desk) to the desk while it runs, so a pod drop
# loses at most a few minutes. Every PERIOD s: progress / state / logs / small rows; each stage group's outputs once its marker is on the
# pod (ref-cf, joined, wraps_snap + qa, strip images, per-wrap img / prev / up / ink/wraps / the stripes' ink); at RU/.final the whole set
# (fields/rsp included) and DIR/.synced, then scripts/pod/score1667.sh DIR (desk CPU, lowprio); then the smoke's outputs until
# RU/.alldone (DIR/.pod_idle: the pod queue is empty). Temp names + rsync --partial; restartable (DIR/.pulled.<group> markers).
# Env: PODH (root@HOST, required), PODPORT (22), PODKEY (~/.ssh/id_ed25519), DIR (/vesuvius/ufsm/podun/PHerc1667-q2pod), RU
# (/root/pod/unroll/PHerc1667), Q (/root/pod/q2), K (/root/ink), PERIOD (180). Run detached: setsid nohup bash q2pull.sh > LOG 2>&1 &
set -u
H=${PODH:?PODH=root@HOST}; SSH="ssh -p ${PODPORT:-22} -i ${PODKEY:-$HOME/.ssh/id_ed25519} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
D=${DIR:-/vesuvius/ufsm/podun/PHerc1667-q2pod}; RU=${RU:-/root/pod/unroll/PHerc1667}; Q=${Q:-/root/pod/q2}; K=${K:-/root/ink}; P=${PERIOD:-180}
R=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd); mkdir -p $D/strip $D/ink $D/fields $D/smoke $D/pod
say() { echo "$(date '+%F %T') q2pull: $*" | tee -a $D/pull.log; }
get() { rsync -a --partial -e "$SSH" "$@" 2>> $D/pull.err; }   # get REMOTE.. LOCALDIR/
has() { $SSH $H "test -e $1" 2> /dev/null; }
grp() { local n=$1 m=$2 dst=$3; shift 3; [ -e $D/.pulled.$n ] && return 0; has $m || return 0
  local a=(); for x in "$@"; do a+=("$H:$x"); done; get "${a[@]}" $dst && touch $D/.pulled.$n && say "pulled $n"; }
say "start: $H -> $D"
while :; do
  if ! $SSH $H true 2> /dev/null; then say "pod unreachable"; sleep $P; continue; fi
  get $H:$RU/progress.log $H:$RU/state $H:$RU/logs $D/ 2> /dev/null
  for f in box.txt axis.json wrapcheck.tsv trim.pap slabs.txt stagetable.tsv; do get $H:$RU/$f $D/ 2> /dev/null; done
  get $H:$Q/podq2.log $H:$Q/status $H:$Q/state $D/pod/ 2> /dev/null; get $H:/root/pod/setup.log $D/pod/ 2> /dev/null
  grp ref $RU/state/ref.done $D/ $RU/ref-cf
  grp join $RU/state/join.done $D/ $RU/joined
  grp wraps $RU/state/snap.done $D/ $RU/wraps_snap $RU/qa
  grp strip $RU/state/stripimg.done $D/strip/ $RU/strip/img_fill $RU/strip/img_pap $RU/strip/distort
  grp stripink $RU/state/stripimg.done $D/ink/ $RU/ink/r3/strip
  grp perwrap $RU/state/perwrap.done $D/ $RU/img $RU/prev $RU/up
  grp wink $RU/state/perwrap.done $D/ink/ $RU/ink/wraps
  if [ ! -e $D/.synced ] && has $RU/.final; then
    get $H:$RU/fields/rsp $D/fields/ && get $H:$RU/ref-cf $H:$RU/joined $H:$RU/wraps_snap $H:$RU/qa $H:$RU/img $H:$RU/prev $H:$RU/up $D/ \
      && get $H:$RU/strip/img_fill $H:$RU/strip/img_pap $H:$RU/strip/distort $D/strip/ && get $H:$RU/ink/wraps $H:$RU/ink/r3/strip $D/ink/ \
      && { touch $D/.synced; say "final pull done: $D/.synced"
           ( CUDA_VISIBLE_DEVICES= ~/bin/memcap ~/bin/lowprio bash $R/scripts/pod/score1667.sh $D PHerc1667-q2pod > $D/score.out 2>&1; say "score: $(tr '\n' ' ' < $D/score/score.txt)" ) & }
  fi
  get $H:$K/podq.log $H:$K/status $H:$K/steptimeA.txt $H:$K/out $D/smoke/ 2> /dev/null
  if has $RU/.alldone || has $Q/state/ALLDONE; then [ -e $D/.synced ] && { touch $D/.pod_idle; say "POD QUEUE EMPTY ($D/.pod_idle)"; wait; exit 0; }; fi
  if has $Q/state/FAILED; then say "pod queue FAILED: $($SSH $H "tail -1 $Q/podq2.log" 2> /dev/null)"; touch $D/.pod_failed; fi
  sleep $P
done

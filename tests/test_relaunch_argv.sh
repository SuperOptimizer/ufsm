#!/usr/bin/env bash
# scripts/rv/train.sh and scripts/univ/u1.train.sh: the trainer's argv is stored at launch (<run>.argv) and <run>.relaunch replays it verbatim
# (REPLAY=1) even when the relaunching shell lacks the launch env (HEADS, EXTRA, ARCH ...): a relaunch that rebuilt the argv from a subset of
# the env restarted a fibre run as the default task (2026-11-20). A stub trainer records its argv and prints the final step line. CPU only.
# Usage: test_relaunch_argv.sh DIR
set -eu
R=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); D=$1
case "$D" in */relaunch*) ;; *) D=$D/relaunch;; esac
rm -rf -- "$D"; mkdir -p "$D"; cd "$D"
cat > stub.sh <<'S'
#!/bin/bash
# the stub trainer: argv (NUL-separated) to $STUB_OUT.<n>, then the last step line so the launcher's loop ends
n=0; while [ -e "$STUB_OUT.$n" ]; do n=$((n + 1)); done; printf '%s\0' "$@" > "$STUB_OUT.$n"
s=0; prev=; for a in "$@"; do [ "$prev" = --steps ] && s=$a; prev=$a; done; echo "step $s loss 0"
S
chmod +x stub.sh; echo '{"sources": []}' > src.json
for L in rv/train.sh univ/u1.train.sh; do
  n=$(basename $(dirname $L)); export STUB_OUT=$D/$n.argv
  env -u HEADS R=$D/runs-$n U=$D/stub.sh SRCJ=$D/src.json GPU=0 LOCK= MINMEM=0 SNAP=2 MEMCAP= HEADS="--dice 0.5 --pos-weight 4" ARCH="--widths 8,16,32" \
      EXTRA="--fib-dir 1" bash $R/scripts/$L fr 4 > $n.launch.log 2>&1 || { echo "FAIL: $L launch"; exit 1; }
  o=$D/runs-$n/fr; [ -s $o.argv ] || { echo "FAIL: $L: no $o.argv"; exit 1; }
  grep -q 'REPLAY=1' $o.relaunch || { echo "FAIL: $L: the relaunch line does not replay"; exit 1; }
  tr '\0' '\n' < $STUB_OUT.0 | grep -qx -- '--fib-dir' || { echo "FAIL: $L: the launch argv lacks --fib-dir"; exit 1; }
  # the relaunch from a shell without the launch env (and the u1 memcap wrapper replaced by the stub's direct call)
  sed -i 's|/home/forrest/bin/memcap -t ||' $o.relaunch
  env -u HEADS -u EXTRA -u ARCH -u STUB_OUT STUB_OUT=$STUB_OUT bash $o.relaunch
  for i in $(seq 1 100); do [ -e $STUB_OUT.1 ] && ! pgrep -f "^bash $R/scripts/$L fr" > /dev/null && break; sleep 0.1; done
  [ -e $STUB_OUT.1 ] || { echo "FAIL: $L: the relaunch did not run the trainer"; cat $o.nohup 2>/dev/null | tail -3; exit 1; }
  cmp -s $STUB_OUT.0 $STUB_OUT.1 || { echo "FAIL: $L: the relaunch argv differs"; diff <(tr '\0' '\n' < $STUB_OUT.0) <(tr '\0' '\n' < $STUB_OUT.1) | head; exit 1; }
  grep -q "replaying $o.argv" $o.resume.log || { echo "FAIL: $L: no replay line"; exit 1; }
done
echo "test_relaunch_argv: ok"

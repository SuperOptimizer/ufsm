#!/usr/bin/env bash
# scripts/run/replay_ab.sh RUN OLD_BIN NEW_BIN [N] (2026-11-14, ops audit #4): does a new binary change the numbers of a live run?
# For a swap window, before the swap. The live gate (gswap.sh) compared 2 training log lines after the swap with 4 before: the
# per-line acc sd is 0.030, so it missed a 0.03 drop and failed a good swap ~3 % of the time, with the schedule confounded.
# This replays the same windows with both binaries:
#   - the run's exact train command, from the last "start" line of RUN.resume.log;
#   - from the same copy of RUN/last.ckpt;
#   - --det 1 (the same batches), --lr 0 --muon-lr 0 (no update: both runs see the same weights at every step);
#   - N steps (default 20).
# Then it compares the per-head losses window by window.
# Identical code paths give identical lines. A kernel or label change shows as a per-head difference:
#   - above TOL (default 0.01 relative on a head's mean) it exits 1 and prints the heads;
#   - that is a code question to settle offline, not on the live run.
# Needs the run's GPUs (stop the trainer first: a swap window). Env: R (run root, /vesuvius/ufsm/v2/lat/runs), TOL, KEEP=1 (keep the dir).
set -u
RUN=${1:?RUN}; OLD=$(readlink -f "${2:?OLD_BIN}"); NEW=$(readlink -f "${3:?NEW_BIN}"); N=${4:-20}; R=${R:-/vesuvius/ufsm/v2/lat/runs}; TOL=${TOL:-0.01}
[ -x "$OLD" ] && [ -x "$NEW" ] || { echo "replay_ab: binaries not executable"; exit 2; }
pgrep -f "^/vesuvius/ufsm/v2/lat/bin/ufsm.* train .* --out $R/$RUN( |$)" > /dev/null && { echo "replay_ab: the $RUN trainer runs: stop it first (swap window)"; exit 2; }
L=$(grep ' start [0-9]* (GPU ' "$R/$RUN.resume.log" | tail -1); [ -n "$L" ] || { echo "replay_ab: no start line in $R/$RUN.resume.log"; exit 2; }
GPU=$(echo "$L" | sed -E 's/.* \(GPU ([^,]*(,[0-9]+)*), lock .*/\1/')
ARGS=$(echo "$L" | sed -E 's/.*\): [^ ]+ train //')   # (the binary path and "train" dropped; the args have no spaces inside values)
W=$(mktemp -d "$R/.replay_ab.XXXXXX"); cp --reflink=auto "$R/$RUN/last.ckpt" "$W/start.ckpt" || exit 2   # (next to the run: reflinks, not tmpfs RAM)
S0=$(head -c 200000 "$W/start.ckpt" | head -1 | grep -o '"step":[0-9]*' | head -1 | cut -d: -f2)
# the run's args without --out / --steps / --resume / --ckpt-every / --val-batches / lr options, then ours
read -ra A <<< "$ARGS"; B=(); skip=0
for a in "${A[@]}"; do
  [ $skip = 1 ] && { skip=0; continue; }
  case $a in --out|--steps|--resume|--ckpt-every|--val-batches|--val-every|--lr|--muon-lr|--det|--warmup|--log-every|--ckpt-ring) skip=1; continue;; esac   # (opt() takes the first: a run's --log-every 50 left 1 logged step in 20, w24 11-27)
  B+=("$a")
done
for v in old new; do
  bin=$OLD; [ $v = new ] && bin=$NEW
  mkdir -p "$W/$v"; cp --reflink=auto "$W/start.ckpt" "$W/$v/last.ckpt"
  echo "replay_ab: $v = $bin, $N windows from step $S0 on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU timeout -k 30 3600 "$bin" train "${B[@]}" --out "$W/$v" --resume "$W/$v/last.ckpt" --steps $((S0 + N)) --det 1 --lr 0 --muon-lr 0 \
      --warmup 0 --ckpt-every 1000000 --ckpt-ring 0 --val-batches 0 --log-every 1 > "$W/$v.log" 2>&1 || { echo "replay_ab: the $v run failed ($W/$v.log)"; tail -5 "$W/$v.log"; exit 2; }
done
python3 - "$W/old.log" "$W/new.log" "$TOL" <<'PY'
import re, sys
heads = ['loss', 'phase bce', 'rv bce', 'normal bce', 'wind4 bce', 'ink bce', 'ink mil', 'fib bce']
def parse(p):
    out = {}
    for l in open(p):
        m = re.match(r'^step +(\d+) ', l)
        if not m: continue
        out[int(m.group(1))] = {h: float(v.group(1)) for h in heads for v in [re.search(r'\b' + re.escape(h) + r' ([0-9.eE+-]+)', l)] if v}
    return out
a, b, tol = parse(sys.argv[1]), parse(sys.argv[2]), float(sys.argv[3])
steps = sorted(set(a) & set(b))
if len(steps) < 5: print('replay_ab: only %d common steps' % len(steps)); sys.exit(2)
same = sum(a[s] == b[s] for s in steps); bad = 0
print('replay_ab: %d windows, %d with identical losses' % (len(steps), same))
for h in heads:
    va = [a[s][h] for s in steps if h in a[s] and h in b[s] and a[s][h] > 0]; vb = [b[s][h] for s in steps if h in a[s] and h in b[s] and a[s][h] > 0]
    if not va: continue
    ma, mb = sum(va) / len(va), sum(vb) / len(vb); r = abs(mb - ma) / max(abs(ma), 1e-9)
    print('  %-11s old %.5f new %.5f  rel %.2e%s' % (h, ma, mb, r, '  > %g' % tol if r > tol else '')); bad += r > tol
print('replay_ab: ' + ('the new binary reproduces the old one\'s losses within %g' % tol if not bad else '%d head(s) changed beyond %g: settle offline before swapping' % (bad, tol)))
sys.exit(1 if bad else 0)
PY
rc=$?
[ "${KEEP:-0}" = 1 ] && echo "replay_ab: kept $W" || case "$W" in */.replay_ab.*) rm -rf -- "$W";; esac
exit $rc

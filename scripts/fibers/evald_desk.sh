#!/usr/bin/env bash
# scripts/fibers/evald_desk.sh STEP...: the desk half of the fibhv-d hold evaluation (scripts/fibers/evald.sh runs on the eval pod EP).
# Per step: starts evald.sh STEP on EP (if not running / done), waits for its done flag, copies the traced lines and logs back, then
# prints per box / family / mode the line statistics (fibre_lens.py: count, metres, median, arc-weighted median, p90, arc share in lines
# >= 1 mm and >= 2 mm) and, on Paris 4 HOLD, the GT judge (fiber-judge --phase against paris4-wind-20261019 phaseA, as fibhv-c's eval).
# Output: /vesuvius/ufsm/fibers/runs/fibhv-d/eval/hold-STEP.desk.log (+ the copied hold-STEP/ dir).
set -u
EP=${EP:-root@213.173.107.10}; EPP=${EPP:-27064}; P="ssh -p $EPP -i $HOME/.ssh/id_ed25519"; D=$(dirname "$(readlink -f "$0")")
C=${C:-/vesuvius/ufsm/fibers/runs/fibhv-d}; U=${U:-/home/forrest/ufsm-fibd/build/ufsm}; mkdir -p $C/eval
J="--um 2.4 --level 2 --slab 12288,512 --region 4096,4608,3584,4352 --tile 256 --d-mm 0.3,0.6,1.2,2.4"
for s in "$@"; do
  o=$C/eval/hold-$s; [ -s $o.desk.log ] && grep -q '^done' $o.desk.log && continue
  # (a pid file, not pgrep: the pattern would match this very ssh command line, 2026-11-11)
  $P $EP "test -e /workspace/fibd/eval/hold-$s/done || { test -s /workspace/fibd/evald-$s.pid && kill -0 \$(cat /workspace/fibd/evald-$s.pid) 2>/dev/null; } || { setsid nohup bash /workspace/ufsm-fibd/scripts/fibers/evald.sh $s > /workspace/fibd/evald-$s.log 2>&1 < /dev/null & echo \$! > /workspace/fibd/evald-$s.pid; }" < /dev/null
  until $P $EP "test -e /workspace/fibd/eval/hold-$s/done" < /dev/null; do sleep 120; done
  mkdir -p $o; rsync -a -e "$P" --include='*/' --include='*.fib' --include='*.log' --include='*.txt' --exclude='*' $EP:/workspace/fibd/eval/hold-$s/ $o/
  { for b in HOLD 1447; do um=2.4; [ $b = 1447 ] && um=8.64
      for f in H V; do for m in tube dyad; do
        echo "== $b $f $m | $(python3 -I $D/fibre_lens.py $o/$b/$f-$m.fib $um)"; grep 'line ends' $o/$b/trace_$f-$m.log
        [ $b = HOLD ] && nice -n 19 $U fiber-judge $o/$b/$f-$m.fib --phase /vesuvius/ufsm/gt/paris4-wind-20261019 phaseA/9.6 $J 2>&1 | grep "$f:"
      done; done
      [ $b = 1447 ] && for f in H V; do echo "-- 1447-HOLD $f mask vs held-out labels"; grep -iE 'thr|dice|prec|recall' $o/1447/eval_$f.txt | head -6; done
    done; echo done; } > $o.desk.log 2>&1
done

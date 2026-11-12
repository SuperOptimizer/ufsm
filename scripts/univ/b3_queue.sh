#!/usr/bin/env bash
# scripts/univ/b3_queue.sh: after b3_1299.sh, redo the PHerc1218 gate (its 2026-11-11 run was killed by systemd-oomd when the dev slice
# filled; the labels are kept, so good.sh goes straight to the gate) with JOBS=8, then the Paris 4 July codes (b1_p4B48.sh).
set -u; H=$(cd "$(dirname "$0")/../.." && pwd); B3=/mnt/scratch/univ/b3
until ! pgrep -f "univ/b3_1299[.]sh" > /dev/null; do sleep 120; done
rmdir $B3/good/PHerc1218.claim 2> /dev/null   # (the dead run's claim; good.sh re-creates its own output dir)
JOBS=8 OUT=$B3/good LAB=$B3/labels Q=$B3/good.log U=$H/build/ufsm TH=10 KEEPFULL=1 nice -n 5 bash $H/scripts/rv/good.sh PHerc1218
bash $H/scripts/univ/b1_p4B48.sh

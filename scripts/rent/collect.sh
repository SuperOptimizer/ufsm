#!/usr/bin/env bash
# scripts/rent/collect.sh: the arms' logs and diagnostics (runs/*.log, runs/*/train.log; checkpoints only with CKPT=1) to the desk under
# /vesuvius/ufsm/dino/ssl/rent/<this host>/. DESK: an ssh destination that reaches the desk from this box (tailscale / port forward), e.g.
#   DESK=forrest@forlindesk2 bash collect.sh        (or run pull.sh on the desk instead, if the desk can reach this box)
set -eu
D=${DESK:?DESK=user@host of the desk}; H=$(hostname)
ssh "$D" mkdir -p /vesuvius/ufsm/dino/ssl/rent/$H
if [ "${CKPT:-0}" = 1 ]; then rsync -a runs/ "$D:/vesuvius/ufsm/dino/ssl/rent/$H/"
else rsync -a --include='*/' --include='*.log' --exclude='*' runs/ "$D:/vesuvius/ufsm/dino/ssl/rent/$H/"; fi
echo "collect: runs/ -> $D:/vesuvius/ufsm/dino/ssl/rent/$H/"

#!/bin/bash
# run one arms.txt line by NAME on GPU G now (outside the queue; claim dir made by the caller): evrun.sh G NAME
D=/vesuvius/ufsm/v2/lat/runs/muonx; G=$1; N=$2
source <(sed -n '/^arm() {/,/^}/p' $D/abq3.sh); U=$D/ufsm.muonx
line=$(awk -v n=$N '$1 == n' $D/arms.txt); set -- $line
echo "$(date +%F_%T) gpu$G start $1 (evrun)" >> $D/done.log
exec 9>/tmp/gpu$G-bench.lock; flock -n 9 || { echo "gpu$G busy"; exit 1; }
arm "$@"

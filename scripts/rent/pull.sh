#!/usr/bin/env bash
# scripts/rent/pull.sh BOX [DIR]: on the desk, pull a rented box's logs (DIR: its kit directory, default ~) into /vesuvius/ufsm/dino/ssl/rent/BOX/
set -eu
B=${1:?ssh destination of the rented box}; R=${2:-~}; O=/vesuvius/ufsm/dino/ssl/rent/${B#*@}; mkdir -p "$O"
rsync -a --include='*/' --include='*.log' --exclude='*' "$B:$R/runs/" "$O/"; echo "pull: $B:$R/runs -> $O"

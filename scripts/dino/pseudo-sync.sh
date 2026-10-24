#!/usr/bin/env bash
# scripts/dino/pseudo-sync.sh [SRC] [HOST:DIR]: copy the pseudo-label boxes (per scan: sources.json, sources2.json, b<i>/codes, b<i>/codes2,
# b<i>/source*.json; not the fields or the CT scratch) to the desk's PSEUDO dir for scripts/universal/full.sh
set -eu

S=${1:-$HOME/ufsm-data/dino/pseudo/scans}; T=${2:-forrest@192.168.0.175:/vesuvius/ufsm/dino/pseudo/scans}
ssh ${T%%:*} mkdir -p ${T#*:}
rsync -a --prune-empty-dirs --include '*/' --include 'sources*.json' --include 'source*.json' --include 'b*/codes/***' --include 'b*/codes2/***' \
  --exclude '*' $S/ $T/
echo "pseudo-sync: $(ls -d $S/*/ | wc -l) scans -> $T"

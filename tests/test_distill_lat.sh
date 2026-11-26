#!/usr/bin/env bash
# train --distill under the latent grid (--lat 4, 2026-11-26): the distill head on decoder level 1 is 4 << 1 = 8 voxels a cell, the
# targets' grid the same (the old 1 << level sized them 64x the head, and the loss read their first 1/64: 1.3 % "cells" in the 11-25
# g256 pair). The wind fixture (256^3 at 9.6 um) with a full feature store (test_distill --write: 8 channels, step 8): 4 steps,
# the head grid checked against the targets (train aborts otherwise), and the logged target share of the head cells must be > 50 %.
# Usage: test_distill_lat.sh UFSM MAKE_WIND_FIXTURE TEST_DISTILL DIR
set -eu
U=$(readlink -f "$1"); MF=$(readlink -f "$2"); TD=$(readlink -f "$3"); D=$4
rm -rf "$D"; mkdir -p "$D"; cd "$D"
fail() { echo "FAIL: $*"; exit 1; }
"$MF" fx 256 2 1 > fx.log 2>&1 || fail "fixture ($D/fx.log)"
"$TD" --write emb 32 32 32 || fail "feature store"
sed 's#"weight": 1,#"weight": 1, "emb": {"groups": ["emb"], "name": "pca", "level": 0},#' fx/sources.json > src-emb.json
grep -q '"emb"' src-emb.json || fail "sources edit"
"$U" train src-emb.json --out run --task unwrap2 --levels 1 --P 128 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 0 \
  --log-every 1 --ckpt-every 1000 --fp4 2 --warmup 0 --seed 5 --widths 16,32,64,64 --lat 4 --lat-guide 3 --sched wsd --cooldown 0 \
  --opt muon --muon-lr 0.002 --lr 0.001 --steps 4 --distill 8 --distill-level 1 --distill-weight 0.3 > train.log 2>&1 || fail "train ($D/train.log)"
grep -q "distill: 8 channels on decoder level 1 (8^3 voxels per cell, latent grid)" train.log || fail "no latent-grid distill line ($D/train.log)"
cov=$(grep '^step' train.log | grep -oE 'distill [0-9.]+ \([0-9.]+% cells' | tail -1 | grep -oE '\(([0-9.]+)%' | tr -d '(%')
[ -n "$cov" ] || fail "no distill loss line ($D/train.log)"
awk -v c=$cov 'BEGIN{exit !(c > 50)}' || fail "distill targets on $cov % of the head cells (want > 50: grid mismatch?)"
echo "test_distill_lat: ok (targets on $cov % of the head cells)"

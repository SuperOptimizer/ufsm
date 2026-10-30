#!/usr/bin/env bash
# scripts/vwt/infer.sh CKPT TILING OUT NPARTS PART:GPU [PART:GPU ...]: pass A of the ray model's whole-volume inference (ufsm vwt-infer)
#   on this host: one vwt-infer lane per PART:GPU (part PART of NPARTS, z bands round robin, on GPU), each restarted with --resume 1 after a
#   crash (at most MAXRESTART times, 30 s apart) until it exits 0. Lanes of other hosts use the same OUT (shared filesystem) or their own
#   OUT dirs (pass B reads the union: a slab's centre is written by exactly one part). Logs: OUT/part-P.log.
# Env: U (binary; default build/ufsm next to this script's repo), ROOT / KEY (the CT; default the P4 9.6 um level), CACHE (z3 cache dir),
#   THREADS (16), MAXRESTART (10), MX8 (1: MX-fp8 conv forward), NICE (5).
# Example (desk, 2 GPUs, parts 0 and 1 of 6): scripts/vwt/infer.sh r1/latest.vwt tiling.bin /vesuvius/ufsm/vwt/infer/ab 6 0:0 1:1
set -u
[ $# -ge 5 ] || { sed -n '2,10p' "$0"; exit 2; }
CKPT=$1; TIL=$2; OUT=$3; NP=$4; shift 4
HERE=$(cd "$(dirname "$0")/../.." && pwd)
U=${U:-$HERE/build/ufsm}
ROOT=${ROOT:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
KEY=${KEY:-PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2}
CACHE=${CACHE:-}; THREADS=${THREADS:-16}; MAXRESTART=${MAXRESTART:-10}; MX8=${MX8:-1}; NICE=${NICE:-5}
[ -x "$U" ] || { echo "no binary $U"; exit 2; }
mkdir -p "$OUT"
lane() {   # PART GPU
  local P=$1 G=$2 n=0 rc
  while :; do
    echo "$(date '+%F %T') part $P/$NP gpu $G start (try $n)" >> "$OUT/part-$P.log"
    nice -n "$NICE" "$U" vwt-infer "$CKPT" "$ROOT" "$KEY" "$OUT" --tiling "$TIL" --part "$P/$NP" --gpu "$G" --threads "$THREADS" \
      --mx8 "$MX8" --resume 1 ${CACHE:+--cache "$CACHE"} >> "$OUT/part-$P.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') part $P/$NP exit $rc" >> "$OUT/part-$P.log"
    [ $rc -eq 0 ] && return 0
    n=$((n + 1)); [ $n -gt "$MAXRESTART" ] && { echo "part $P: giving up after $n tries" >> "$OUT/part-$P.log"; return 1; }
    sleep 30
  done
}
pids=()
for pg in "$@"; do lane "${pg%%:*}" "${pg##*:}" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc

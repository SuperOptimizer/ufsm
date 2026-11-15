#!/usr/bin/env bash
# fith2_prof.sh [GPU]: post-pass GPU work for audit unroll #9 (the fused --fit-h 2 fit-field kernel, UFSM_PRED_FIT_FUSED_SMALLH=1,
# default off). Needs a GPU window (desk: a scratch2 swap window; or a pod). ~10 min on one 16 GB card. Writes only under $OUT.
#   1. test_fit_lat (build-verify of this tree): the fused kernel at h 8 / 4 / 2 against the plane path, every accumulator bit for bit,
#      then UFSM_TEST_BENCH=1: plane path vs fused ms on a 256^3 window at h 8 and h 2
#   2. the production fields predict (unroll.sh fpred1: latent u1, window 640 / 64, --fit-h 2) on a 1024 x 1024 x 1024 box of 1667, with
#      UFSM_PRED_PROF=1, plane path vs UFSM_PRED_FIT_FUSED_SMALLH=1: wall, the "fit:" ticks, and every field shard byte-compared
#      ("IDENTICAL" expected: the kernel sums in the plane path's order)
# Env: U (binary, default build/ufsm of this tree), CK (default the 1667 unroll's u1 snap), R K UM LV (CT), BOX (z,y,x,nz,ny,nx level-LV
# voxels), AXIS, OUT. Verdict lines in $OUT/summary.txt; decide the default from them (DESIGN "Shell-script and coordinate ...").
set -u
G=${1:-0}; REPO=$(cd "$(dirname "$0")/../.." && pwd); O=${OUT:-/vesuvius/ufsm/tmp-fith2/gpu-$(date +%m%d-%H%M)}; mkdir -p "$O"
U=${U:-$REPO/build/ufsm}; RUN=/vesuvius/ufsm/unroll/PHerc1667-u1
CK=${CK:-/vesuvius/ufsm/v2/lat/runs/u1/snap/13000.ckpt}; R=${R:-/vesuvius/ufsm/v2/ct}; K=${K:-PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr}
UM=${UM:-2.399}; LV=${LV:-2}; BOX=${BOX:-1536,1024,1024,1024,1024,1024}; AXIS=${AXIS:-$RUN/axis.json}
export CUDA_VISIBLE_DEVICES=$G
S=$O/summary.txt; : > "$S"
say() { echo "$*" | tee -a "$S"; }
say "fith2_prof: GPU $G, U $U, CK $CK, box $BOX, out $O, $(date)"
for f in "$U" "$CK" "$AXIS"; do [ -e "$f" ] || { say "missing $f"; exit 1; }; done
# 1. kernel identity + bench
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  T=$REPO/build-verify/test_fit_lat; [ -x "$T" ] || { say "no $T (cmake --build build-verify --target test_fit_lat)"; exit 1; }
  UFSM_TEST_BENCH=1 timeout -k 10 600 "$T" > "$O/test_fit_lat.log" 2>&1; say "test_fit_lat rc $? ($(tail -1 "$O/test_fit_lat.log"))"
  grep -E '^FAIL|^bench' "$O/test_fit_lat.log" | sed 's/^/  /' | tee -a "$S"
fi
# 2. the production predict, plane path vs fused
pred() {   # pred NAME [env..]
  local n=$1; shift; rm -rf "${O:?}/$n" "${O:?}/$n.tmp"
  env UFSM_PRED_PROF=1 "$@" /usr/bin/time -f "wall %e s, max RSS %M KB" timeout -k 10 1800 "$U" predict "$CK" "$R" "$K" "$O/$n.tmp" --um "$UM" --level "$LV" --box "$BOX" \
    --window 640 --halo 64 --shard 512 --grid-origin 0,0,0 --ema 1 --q 8 --threads ${PTH:-14} --gpu 0 --axis "$AXIS" --levels 1 \
    --fit-zarr "$O/$n" --fit-h 2 --fit-q 8 --fit-q-phase 4 --no-out 1 > "$O/$n.log" 2>&1
  local rc=$?; rm -rf "${O:?}/$n.tmp"
  say "$n: rc $rc, $(tail -1 "$O/$n.log")"
  grep -E 'fused|plane path|fit: |prof|windows' "$O/$n.log" | tail -12 | sed 's/^/    /' >> "$S"
  return $rc
}
pred plane UFSM_PRED_FIT_FUSED_SMALLH=0 && pred fused UFSM_PRED_FIT_FUSED_SMALLH=1 || { say "a predict failed"; exit 1; }
grep -q 'fused from the cell logits' "$O/fused.log" || say "WARNING: the fused run did not take the fused kernel (see fused.log)"
nd=0; nf=0
while IFS= read -r f; do nf=$((nf + 1)); cmp -s "$O/plane/$f" "$O/fused/$f" || { nd=$((nd + 1)); [ $nd -le 5 ] && say "  differs: $f"; }; done < <(cd "$O/plane" && find . -type f ! -name '*.json' | sort)
[ "$(cd "$O/plane" && find . -type f | wc -l)" = "$(cd "$O/fused" && find . -type f | wc -l)" ] || { say "file counts differ"; nd=$((nd + 1)); }
[ $nd = 0 ] && say "IDENTICAL plane fused ($nf field files)" || say "DIFFER plane fused ($nd of $nf files)"
say "done $(date)"

#!/usr/bin/env bash
# align_ab.sh CKPT ROOT CT-GROUP UM BOX OUT (data audit #9, datafix 2026-11-14; post-pass GPU job): predict on one box twice with the
# same window / halo, once on the default grid (window origins box + k stride - halo) and once with --align 128 (origins on the 128
# grid like training's snapped windows), then compare the two outputs voxel by voxel over the box.
#   BOX: z,y,x,nz,ny,nx on the CT level (pick an origin NOT = halo mod 128, e.g. ...+64, so the default grid is misaligned).
#   Env: LEVEL (2), WIN (640), HALO (64; window - 2 halo must be a multiple of 128), GPU (0), U (build/ufsm), EXTRA (more predict flags,
#   e.g. "--heads 1"), LOCK (a flock file to hold for the GPU part; empty = none).
# Output: OUT/{grid,align} predictions, OUT/ab.txt: mean |a - b| (uint8), the share of voxels differing by > 8 and > 32, the Pearson
# correlation, and the same within 64 voxels of a window edge vs the interior (edge effects vs a global shift). Decision rule (DESIGN):
# mean |diff| < 1 and > 32 share < 0.1 % = alignment does not matter, keep the default; else make --align 128 the default.
set -eu
CK=$1 ROOT=$2 CT=$3 UM=$4 BOX=$5 OUT=$6
U=${U:-$(cd "$(dirname "$0")/../.." && pwd)/build/ufsm}; LEVEL=${LEVEL:-2}; WIN=${WIN:-640}; HALO=${HALO:-64}; GPU=${GPU:-0}
mkdir -p "$OUT"
run() { local tag=$1; shift
  if [ -n "${LOCK:-}" ]; then flock "$LOCK" "$U" predict "$CK" "$ROOT" "$CT" "$OUT/$tag" --um "$UM" --level "$LEVEL" --box "$BOX" --window "$WIN" --halo "$HALO" --gpu "$GPU" ${EXTRA:-} "$@"
  else "$U" predict "$CK" "$ROOT" "$CT" "$OUT/$tag" --um "$UM" --level "$LEVEL" --box "$BOX" --window "$WIN" --halo "$HALO" --gpu "$GPU" ${EXTRA:-} "$@"; fi > "$OUT/$tag.log" 2>&1; }
[ -d "$OUT/grid" ] || run grid
[ -d "$OUT/align" ] || run align --align 128
LV=$(ls "$OUT/grid" | grep -E '^[0-9.]+$' | head -1)
IFS=, read -r bz by bx nz ny nx <<< "$BOX"
"$U" read "$OUT/grid" "$LV" 0 0 0 "$nz" "$ny" "$nx" "$OUT/grid.raw" > /dev/null
"$U" read "$OUT/align" "$LV" 0 0 0 "$nz" "$ny" "$nx" "$OUT/align.raw" > /dev/null
python3 -I - "$OUT" "$nz" "$ny" "$nx" "$WIN" "$HALO" "$bz" "$by" "$bx" <<'PY' | tee "$OUT/ab.txt"
import sys, numpy as np
o, nz, ny, nx, W, H, bz, by, bx = sys.argv[1], *map(int, sys.argv[2:])
a = np.fromfile(o + '/grid.raw', np.uint8).reshape(nz, ny, nx).astype(np.float32)
b = np.fromfile(o + '/align.raw', np.uint8).reshape(nz, ny, nx).astype(np.float32)
d = np.abs(a - b); S = W - 2 * H
def edge(n, b0, origin0):   # distance (voxels) to the nearest window-interior edge of a grid with origins origin0 + k S
    p = (np.arange(n) + b0 - origin0) % S; return np.minimum(p, S - p)
# default grid: interiors start at the box origin (+ k S); aligned grid: at the 128-aligned k S
e = np.minimum.reduce(np.meshgrid(*[np.minimum(edge(n, 0, 0), edge(n, b0, 0)) for n, b0 in ((nz, bz), (ny, by), (nx, bx))], indexing='ij'))
for tag, m in (('all', np.ones_like(d, bool)), ('near a window edge (< 64)', e < 64), ('interior (>= 64)', e >= 64)):
    if not m.any(): continue
    r = np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1] if a[m].std() > 0 and b[m].std() > 0 else float('nan')
    print('%-28s voxels %11d  mean|diff| %.3f  >8 %.4f%%  >32 %.4f%%  corr %.5f' % (tag, m.sum(), d[m].mean(), 100 * (d[m] > 8).mean(), 100 * (d[m] > 32).mean(), r))
PY

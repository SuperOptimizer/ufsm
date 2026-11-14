# sourced by the shell tests whose byte-identity holds only under UFSM_STRICT_MATH (tests/fm_tol.h, DESIGN 2026-11-14 "Fast math
# everywhere"). CMake writes $BUILD/strict_math (1: -DUFSM_STRICT_MATH=ON, 0: the default fast-math build); B is the suite's build
# dir (run_suite.sh exports it; build-verify by hand).
fm_strict() { [ "$(cat "${B:-build-verify}/strict_math" 2> /dev/null)" = 1 ]; }
# fm_cmp_u8 A B MAXD FRAC [WHAT]: two files of u8 voxels (same length) agree exactly under strict math; else every byte within
# +-MAXD and at most ceil(FRAC * n) bytes differ. Prints the counts; returns 0 on pass.
fm_cmp_u8() {
    local a=$1 b=$2 maxd=$3 frac=$4 what=${5:-$1}
    if fm_strict; then cmp -s "$a" "$b" && return 0; echo "FAIL $what: not byte-identical [strict]" >&2; return 1; fi
    python3 -I - "$a" "$b" "$maxd" "$frac" "$what" << 'EOF'
import math, sys
import numpy as np
a = np.fromfile(sys.argv[1], np.uint8); b = np.fromfile(sys.argv[2], np.uint8); maxd = int(sys.argv[3]); frac = float(sys.argv[4])
if a.size != b.size: print(f"FAIL {sys.argv[5]}: sizes {a.size} vs {b.size}", file=sys.stderr); sys.exit(1)
d = np.abs(a.astype(np.int16) - b.astype(np.int16)); nd = int((d > 0).sum()); allow = math.ceil(frac * a.size) if frac > 0 else 0
worst = int(d.max()) if a.size else 0; bad = nd > allow or worst > maxd
if nd: print(f"{'FAIL' if bad else 'ok'} {sys.argv[5]}: {nd} of {a.size} bytes differ (allowed {allow}), worst {worst} (allowed {maxd})", file=sys.stderr)
sys.exit(1 if bad else 0)
EOF
}
# fm_det_cmp A.log B.log WHAT: det_cmp (tests/det_cmp.sh, source it first) under strict math; otherwise two --det runs whose label
# planes come from GPU vs CPU code: the device inputs (dettrace, labels cut) equal, the per-step losses within 2 %, no batch hashes.
fm_det_cmp() {
  if fm_strict; then det_cmp "$@"; return; fi
  local a=$1 b=$2 what=$3 xa xb la lb
  xa=$(grep -E '^dettrace' "$a" | sed -E 's/ labels .*//'); xb=$(grep -E '^dettrace' "$b" | sed -E 's/ labels .*//')
  [ -n "$xa" ] && [ "$xa" = "$xb" ] || { echo "FAIL: $what: no / differing device inputs"; return 1; }
  la=$(grep -E '^step' "$a" | awk '{print $2, $6}'); lb=$(grep -E '^step' "$b" | awk '{print $2, $6}')
  [ -n "$la" ] && [ "$(echo "$la" | wc -l)" = "$(echo "$lb" | wc -l)" ] || { echo "FAIL: $what: no / unequal step lines"; return 1; }
  paste -d' ' <(echo "$la") <(echo "$lb") | awk '{d = $2 - $4; if (d < 0) d = -d; if ($1 != $3 || d > 0.02 * ($2 < 0 ? -$2 : $2) + 1e-6) bad = 1} END {exit bad}' \
    || { echo "FAIL: $what: losses differ by more than 0.02"; return 1; }
  DET_CMP_N="$(grep -c '^batchhash' "$a") (hashes not compared: fast math)"
}

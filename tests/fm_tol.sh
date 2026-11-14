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

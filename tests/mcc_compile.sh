#!/bin/bash
# mcc_compile.sh: every host .c and every CUDA unit compiled by mcc (microc), so that a C / C++ idiom mcc does not take yet shows up at
# merge time, not at the next toolchain sweep (mcc reads .cu as its own C++ subset: two units from 11-13 / 11-14 broke it unnoticed).
# Compile only (objects into a temp dir), no link, no GPU. The flags are cmake/mcc.cmake's (CMakeLists.txt, UFSM_TOOLCHAIN=mcc).
#   UFSM_MCC       the mcc binary (default: mcc on PATH, else $MCC_HOME/mcc); none: the test is skipped (passes)
#   UFSM_MCC_CUDA  mcudahw's directory (default: <mcc's directory>/cudahw); its include/ holds mcuda_device.h and the CUDA headers
#   UFSM_NCPU      parallel compiles (default 4)
set -u
cd "$(dirname "$0")/.." || exit 2
M=${UFSM_MCC:-$(command -v mcc || true)}
[ -z "$M" ] && [ -n "${MCC_HOME:-}" ] && [ -x "$MCC_HOME/mcc" ] && M=$MCC_HOME/mcc
if [ -z "$M" ] || [ ! -x "$M" ]; then echo "mcc_compile: skipped (no mcc: set UFSM_MCC or MCC_HOME, or put mcc on PATH)"; exit 0; fi
H=${UFSM_MCC_CUDA:-$(dirname "$M")/cudahw}/include
if [ ! -f "$H/mcuda_device.h" ]; then echo "mcc_compile: no mcudahw headers at $H (microc: sh mkcuda.sh MCC DIR hw; or UFSM_MCC_CUDA)"; exit 1; fi
T=$(mktemp -d "${TMPDIR:-/tmp}/mcc_compile.XXXXXX") || exit 2
case "$T" in "${TMPDIR:-/tmp}"/mcc_compile.*) trap 'rm -rf -- "$T"' EXIT;; *) echo "mcc_compile: odd temp dir $T"; exit 2;; esac
# the strict units (no fast math, -fmad=false), as CMakeLists.txt lists them: only under UFSM_STRICT_MATH=1 (the CMake option's
# build); by default every unit is --cuda-fast-math and the host -ffast-math -fno-finite-math-only (DESIGN 2026-11-14 "Fast math
# everywhere ..."; never -ffinite-math-only: it lets mcc fold the NaN guards)
STRICT=$(sed -n 's/^set(UFSM_STRICT_UNITS \(.*\))$/\1/p' CMakeLists.txt)
[ -n "$STRICT" ] || { echo "mcc_compile: UFSM_STRICT_UNITS not found in CMakeLists.txt"; exit 2; }
SM=${UFSM_STRICT_MATH:-0}; case $SM in ON|on|1) SM=1;; *) SM=0;; esac
if [ $SM = 1 ]; then HFP=; else HFP="-ffast-math -fno-finite-math-only"; fi
# the device library's function names (mcudahw's hw_lib table, cudahw/obj/lib.c), for the link check in one()
LIB=; LC=${UFSM_MCC_CUDA:-$(dirname "$M")/cudahw}/obj/lib.c
if [ -f "$LC" ]; then LIB=$T/lib.names; sed -n 's/^ *{"\(__mcuda_[A-Za-z0-9_]*\)".*/\1/p' "$LC" | sort -u > "$LIB"; [ -s "$LIB" ] || LIB=; fi
[ -n "$LIB" ] || echo "mcc_compile: no device library table at $LC: the link check is skipped"
export M H T STRICT SM HFP LIB
one() {   # one SOURCE: compile it, print "FAIL name: first error" on failure
  f=$1; b=$(basename "${f%.*}")
  case $f in
    *.cu) case $b in lp_*) a=--cuda-arch=1200a;; *) a=--cuda-arch=1200;; esac
          fp=--cuda-fast-math; [ $SM = 1 ] && [[ " $STRICT " == *" $b "* ]] && fp=-fmad=false
          set -- "$M" -mglibc -O3 $a $fp -D_GNU_SOURCE -DUFSM_ARCH=120 -DUFSM_CUDA_VERSION=mcc -isystem "$H" -include "$H/mcuda_device.h" -Isrc;;
    *)    set -- "$M" -mglibc -std=c23 -O3 -march=native $HFP -pthread -D_GNU_SOURCE -Isrc -Ithird_party -Ithird_party/surfcomp -isystem "$H";;
  esac
  case $f in *.cu) set -- "$@" --ptx-out="$T/$b.ptx";; esac
  if ! "$@" -c "$f" -o "$T/$b.o" > "$T/$b.log" 2>&1; then echo "FAIL $b: $(grep -m1 -E 'error|fatal' "$T/$b.log" || head -1 "$T/$b.log")"; return; fi
  # link check: every device function the PTX calls but does not define must be in mcudahw's device library (the table its
  # loader resolves __mcuda_NAME against; __mcuda_printf the driver's). An undefined one fails only when the module loads (11-15:
  # __fdividef / __frcp_rn / __threadfence_system as implicit externs, in most lp kernels); ptxas -c and this compile miss it
  if [ -f "$T/$b.ptx" ] && [ -n "$LIB" ]; then
    u=$(sed -n 's/^\.extern \.func.* \(__mcuda_[A-Za-z0-9_]*\) *(.*/\1/p' "$T/$b.ptx" | sort -u | grep -vxF -f "$LIB" | grep -vx '__mcuda_printf' | head -3 | tr '\n' ' ')
    [ -n "$u" ] && echo "FAIL $b: device functions no library defines (the module would not load): $u"
  fi
}
export -f one
out=$(ls src/*.c src/*.cu third_party/surfcomp/surfcomp.c | xargs -P "${UFSM_NCPU:-4}" -I{} bash -c 'one "$@"' _ {})
n=$(ls src/*.c src/*.cu third_party/surfcomp/surfcomp.c | wc -l)
if [ -n "$out" ]; then echo "$out"; echo "mcc_compile: $(echo "$out" | wc -l) of $n units failed ($M)"; exit 1; fi
echo "mcc_compile: ok ($n units, $M${LIB:+; device calls all in the library})"

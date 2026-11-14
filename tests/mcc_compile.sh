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
# the strict units (no fast math, -fmad=false), as CMakeLists.txt lists them
STRICT=$(sed -n 's/^set(UFSM_STRICT_UNITS \(.*\))$/\1/p' CMakeLists.txt)
[ -n "$STRICT" ] || { echo "mcc_compile: UFSM_STRICT_UNITS not found in CMakeLists.txt"; exit 2; }
export M H T STRICT
one() {   # one SOURCE: compile it, print "FAIL name: first error" on failure
  f=$1; b=$(basename "${f%.*}")
  case $f in
    *.cu) case $b in lp_*) a=--cuda-arch=1200a;; *) a=--cuda-arch=1200;; esac
          fp=--cuda-fast-math; [[ " $STRICT " == *" $b "* ]] && fp=-fmad=false
          set -- "$M" -mglibc -O3 $a $fp -D_GNU_SOURCE -DUFSM_ARCH=120 -DUFSM_CUDA_VERSION=mcc -isystem "$H" -include "$H/mcuda_device.h" -Isrc;;
    *)    set -- "$M" -mglibc -std=c23 -O3 -march=native -pthread -D_GNU_SOURCE -Isrc -Ithird_party -Ithird_party/surfcomp -isystem "$H";;
  esac
  if ! "$@" -c "$f" -o "$T/$b.o" > "$T/$b.log" 2>&1; then echo "FAIL $b: $(grep -m1 -E 'error|fatal' "$T/$b.log" || head -1 "$T/$b.log")"; fi
}
export -f one
out=$(ls src/*.c src/*.cu third_party/surfcomp/surfcomp.c | xargs -P "${UFSM_NCPU:-4}" -I{} bash -c 'one "$@"' _ {})
n=$(ls src/*.c src/*.cu third_party/surfcomp/surfcomp.c | wc -l)
if [ -n "$out" ]; then echo "$out"; echo "mcc_compile: $(echo "$out" | wc -l) of $n units failed ($M)"; exit 1; fi
echo "mcc_compile: ok ($n units, $M)"

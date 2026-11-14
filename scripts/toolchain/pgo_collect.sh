#!/usr/bin/env bash
# pgo_collect.sh WORKLOAD [BUILD_DIR] [PGO_DIR]: the profile for UFSM_PGO=use (DESIGN 2026-11-14 "Fast math everywhere ...").
#   1. configures BUILD_DIR (default build-pgo) as the release preset with UFSM_PGO=gen UFSM_PGO_DIR=PGO_DIR/raw (instrumented host
#      C; the device code is not instrumented), builds ufsm (~/bin/lowprio, -j 8: the desk rule; JOBS overrides);
#   2. runs every non-comment line of WORKLOAD under bash, in order (a failing line stops it), with U = the instrumented ufsm,
#      B = BUILD_DIR (its test binaries are instrumented too: their profiles key the same CORE functions), MF = B/make_wind_fixture,
#      W = PGO_DIR/work (scratch, emptied first); lines run in the repo root. The
#      runs whose host time matters (sampler workers, predict / fit / render / count-field CPU paths). Instrumented code runs
#      ~1.5-3x slower on the host; keep each run short (minutes);
#   3. merges PGO_DIR/raw/*.profraw into PGO_DIR/ufsm.profdata (llvm-profdata of the clang that built it).
# Then: cmake --preset release -DUFSM_PGO=use -DUFSM_PGO_DIR=PGO_DIR (default PGO_DIR = <repo>/pgo, the CMake default).
# A profile stays usable as sources change (stale functions just lose it); recollect after large host-code changes.
set -euo pipefail
WL=$(readlink -f "${1:?usage: pgo_collect.sh WORKLOAD [BUILD_DIR] [PGO_DIR]}")
H=$(cd "$(dirname "$0")/../.." && pwd); cd "$H"
BD=${2:-build-pgo}; PD=$(readlink -m "${3:-$H/pgo}"); JOBS=${JOBS:-8}
LP=; [ -x ~/bin/lowprio ] && LP=~/bin/lowprio
case "$PD" in /?*/*) ;; *) echo "pgo_collect: refusing PGO_DIR '$PD'" >&2; exit 2;; esac
rm -rf -- "$PD/raw"; mkdir -p "$PD/raw"
cmake --preset release -B "$BD" -DUFSM_PGO=gen -DUFSM_PGO_DIR="$PD/raw" > "$PD/configure.log"
TG=${PGO_TARGETS:-ufsm make_wind_fixture test_countfield test_fit test_render_stream test_volcomp test_volcomp_fast}
$LP cmake --build "$BD" -j "$JOBS" --target $TG > "$PD/build.log"
rm -rf -- "$PD/work"; mkdir -p "$PD/work"
export U=$H/$BD/ufsm B=$H/$BD MF=$H/$BD/make_wind_fixture W=$PD/work LLVM_PROFILE_FILE="$PD/raw/ufsm-%p-%m.profraw"
n=0
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n + 1)); echo "pgo_collect: [$n] $line"
  bash -c "$line"
done < "$WL"
PROF=$(ls -d "$(dirname "$(readlink -f "$(grep -m1 '^CMAKE_C_COMPILER:' "$BD/CMakeCache.txt" | cut -d= -f2)")")"/llvm-profdata 2> /dev/null || command -v llvm-profdata)
"$PROF" merge -o "$PD/ufsm.profdata" "$PD"/raw/*.profraw
echo "pgo_collect: $PD/ufsm.profdata from $(ls "$PD"/raw/*.profraw | wc -l) runs of $n workload lines"

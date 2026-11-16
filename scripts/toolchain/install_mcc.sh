#!/bin/bash
# install_mcc.sh [PREFIX]: the microc toolchain ufsm builds with (cmake/mcc.cmake), at the revision in cmake/mcc.version, installed into
# PREFIX (default ~/.local/mcc): PREFIX/mcc (the compiler, its glibc-mode headers and libmcc.a beside it), PREFIX/cudahw (mcudahw:
# include/ and lib64/, the CUDA runtime on NVIDIA's driver). Needs git and a C compiler for stage 0 (cc / gcc / clang: CC=...);
# mcc's glibc-mode link runs /usr/bin/cc (MCC_CC=/full/path/clang-22 where there is none). About a minute on one core.
#   MICROC_URL   the repository (default https://github.com/SuperOptimizer/microc)
#   MICROC_REV   a revision other than cmake/mcc.version's
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
P=${1:-$HOME/.local/mcc}
rev=${MICROC_REV:-$(tr -d ' \n' < "$here/cmake/mcc.version")}
url=${MICROC_URL:-https://github.com/SuperOptimizer/microc}
W=$(mktemp -d "${TMPDIR:-/tmp}/install_mcc.XXXXXX")
case "$W" in "${TMPDIR:-/tmp}"/install_mcc.*) trap 'rm -rf -- "$W"' EXIT;; *) echo "install_mcc: odd temp dir $W"; exit 2;; esac
if [ -f "$P/VERSION" ] && [ "$(cat "$P/VERSION")" = "$rev" ] && [ -x "$P/mcc" ]; then echo "install_mcc: $P is at $rev already"; exit 0; fi
echo "install_mcc: microc $rev -> $P"
git clone -q "$url" "$W/microc"
cd "$W/microc"
git checkout -q "$rev"
CCX=${CC:-$(command -v cc || command -v gcc || command -v clang || true)}
[ -n "$CCX" ] || { echo "install_mcc: no C compiler for stage 0 (set CC)"; exit 1; }
$CCX -std=c89 -O2 -w -o build-mcc src/*.c 2> "$W/stage0.log" || { cat "$W/stage0.log"; exit 1; }
mkdir -p build/stage0 && mv build-mcc build/stage0/mcc
sh mklxlibc.sh build/stage0/mcc build/stage0 x86_64 > "$W/libc.log" 2>&1 || { tail -20 "$W/libc.log"; exit 1; }
HWCC="build/stage0/mcc -mglibc" sh mkcuda.sh build/stage0/mcc build/stage0 hw > "$W/cuda.log" 2>&1 || { tail -20 "$W/cuda.log"; exit 1; }
# a smoke test before replacing anything: a threaded C23 program through the glibc link
printf '#include <stdio.h>\n#include <pthread.h>\nstatic void *f(void *a) { return a; }\nint main(void) { pthread_t t; void *r = nullptr; pthread_create(&t, nullptr, f, (void *)42); pthread_join(t, &r); printf("%%ld\\n", (long)r); return 0; }\n' > "$W/t.c"
build/stage0/mcc -mglibc -std=c23 -pthread "$W/t.c" -o "$W/t" && [ "$("$W/t")" = 42 ] || { echo "install_mcc: smoke test failed"; exit 1; }
N="$P.new.$$"
mkdir -p "$N"
cp -a build/stage0/mcc build/stage0/glibc build/stage0/cudahw "$N/"
[ -d build/stage0/linux-x64 ] && cp -a build/stage0/linux-x64 "$N/"
echo "$rev" > "$N/VERSION"
if [ -e "$P" ]; then
  case "$P" in */mcc|*/mcc/) ;; *) echo "install_mcc: refusing to replace $P (the prefix must end in /mcc)"; exit 1;; esac
  O="$P.old.$$"; mv -- "$P" "$O"; mv -- "$N" "$P"; rm -rf -- "$O"
else
  mkdir -p "$(dirname "$P")"; mv -- "$N" "$P"
fi
echo "install_mcc: ok ($P/mcc, $P/cudahw; microc $rev)"

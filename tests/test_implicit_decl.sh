#!/usr/bin/env bash
# tests/test_implicit_decl.sh: no implicit function declarations / implicit int in src/*.c and tests/*.c (lead 11-27). mcc accepts them
# silently and miscompiles the call (phasecmp.c on w11prep called floor / ceil / llround without math.h: garbage, no diagnostic; fixed
# b32c16ca), so clang -fsyntax-only -Werror=implicit-function-declaration -Werror=implicit-int is the gate until mcc rejects them in C23
# mode. Only those diagnostics count (other clang errors, e.g. CUDA-only headers, are not this test's business). A canary with a
# missing <math.h> must be caught, so a clang that stops reporting it cannot pass silently. CPU only, parallel, ~10-20 s; SKIP without clang.
set -u
H=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); cd "$H" || exit 1
CC=${CLANG:-clang}
command -v "$CC" > /dev/null || { echo "SKIP: no clang (set CLANG)"; exit 0; }
D=$(mktemp -d "${TMPDIR:-/tmp}/test_implicit_decl_XXXXXX"); trap 'rm -rf -- "$D"' EXIT
FL=(-std=c2x -fsyntax-only -w -Werror=implicit-function-declaration -Werror=implicit-int -D_GNU_SOURCE
    -Isrc -Itests -Ithird_party -Ithird_party/surfcomp -I/usr/local/cuda/include)
PAT='implicit (declaration|int)|call to undeclared|type specifier missing'
printf 'int main(void) { double x = 2.5; return (int)floor(x) + (int)llround(x); }\n' > "$D/canary.c"
"$CC" "${FL[@]}" "$D/canary.c" 2>&1 | grep -qE "$PAT" || { echo "FAIL: clang did not flag the canary's missing <math.h> ($CC ${FL[*]})"; exit 1; }
ls src/*.c tests/*.c | xargs -P "${UFSM_NCPU:-4}" -n 8 "$CC" "${FL[@]}" > "$D/out" 2>&1
hits=$(grep -E "$PAT" "$D/out")
n=$(ls src/*.c tests/*.c | wc -l)
if [ -n "$hits" ]; then echo "FAIL: implicit declarations in $(echo "$hits" | cut -d: -f1 | sort -u | wc -l) files:"; echo "$hits" | head -n 40; exit 1; fi
echo "PASS: $n C files, no implicit function declaration / implicit int"

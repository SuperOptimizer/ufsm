#!/usr/bin/env bash
# test_unroll_cfrobust.sh (2026-11-27): the unroll ref's count-field integrates in L1 by default (--cf-robust 8 --cf-robust-delta 0.02,
# the 1667 region A/B), CFROBUST= restores the L2 ref, CFROBUST="..." replaces the flags, REFX still adds its own; CFROBUST is in the
# run's recorded env list. refpool() is taken from unroll.sh itself. CPU, < 1 s.
set -u
cd "$(dirname "$0")/.." || exit 1
F=scripts/unroll/unroll.sh; fails=0; ck() { if eval "$2"; then echo "ok: $1"; else echo "FAIL: $1"; fails=$((fails + 1)); fi; }
def=$(grep -E '^refpool\(\) \{' $F)
ck "refpool found" '[ -n "$def" ]'
run() { env -i PATH="$PATH" bash -c "$def
$1" 2>/dev/null; }
ck "default: --cf-robust 8 --cf-robust-delta 0.02" '[[ " $(run refpool) " == *" --cf-robust 8 --cf-robust-delta 0.02 "* ]]'
ck "CFROBUST= : no --cf-robust (L2)" '[[ " $(run "CFROBUST=; refpool") " != *"--cf-robust"* ]]'
ck "CFROBUST=\"--cf-robust 4\": replaced" '[[ " $(run "CFROBUST=\"--cf-robust 4\"; refpool") " == *" --cf-robust 4 "* ]] && [[ " $(run "CFROBUST=\"--cf-robust 4\"; refpool") " != *"--cf-robust 8"* ]]'
ck "REFX still appended" '[[ " $(run "REFX=\"--rho-scale 2\"; refpool") " == *" --rho-scale 2 "* ]]'
ck "CFROBUST recorded in the run env" 'grep -q " REFX CFROBUST REFCMD " $F'
[ $fails = 0 ] && echo "test_unroll_cfrobust: ok" || echo "test_unroll_cfrobust: $fails failure(s)"
exit $((fails != 0))

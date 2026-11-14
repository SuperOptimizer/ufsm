#!/usr/bin/env bash
# tests/sanitize.sh TOOL CMD...: CMD under compute-sanitizer --tool TOOL (racecheck: shared-memory hazards; synccheck: barrier misuse;
# memcheck / initcheck), child processes included (the shell tests start ufsm). Any report fails the test (exit 86). SKIP without the
# tool. The slow tier's sanitize group (ops audit #9). Stream-ordering races between kernels (global memory) are not what racecheck
# sees: tests/test_serialize.sh (UFSM_SERIALIZE=1 against the default) covers those.
set -u
T=${1:?TOOL}; shift
CS=${COMPUTE_SANITIZER:-$(command -v compute-sanitizer 2> /dev/null || echo /usr/local/cuda/bin/compute-sanitizer)}
[ -x "$CS" ] || { echo "SKIP: no compute-sanitizer"; exit 0; }
exec "$CS" --tool "$T" --error-exitcode 86 --target-processes all --print-limit 20 "$@"

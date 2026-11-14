#!/bin/bash
# suite_check.sh SRCDIR "BIN;BIN;...": the suite binaries (ufsm_exe ... SUITE in tests/targets.cmake) against tests/suite.txt.
# A binary that no manifest line runs, or a manifest line whose $B/binary has no target, is an error (a merge that added one half).
cd "$1" || exit 2
IFS=';' read -ra bins <<< "$2"
declare -A have; for b in "${bins[@]}"; do have[$b]=1; done
declare -A used
while IFS= read -r b; do used[$b]=1; done < <(grep -v '^\s*#' tests/suite.txt | grep -oE '\$B/[A-Za-z0-9_]+' | sed 's|\$B/||' | sort -u)
rc=0
for b in "${!used[@]}"; do [ -z "${have[$b]:-}" ] && { echo "suite-check: tests/suite.txt runs \$B/$b, but no ufsm_exe($b SUITE ...) in tests/targets.cmake"; rc=1; }; done
ind=(); for b in "${bins[@]}"; do [ -z "${used[$b]:-}" ] && ind+=("$b"); done
[ $rc = 0 ] && echo "suite-check: ok (${#bins[@]} suite binaries; ${#ind[@]} not named as \$B/x in tests/suite.txt: helpers / Python-run)"
exit $rc

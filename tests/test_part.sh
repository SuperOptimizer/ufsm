# tests/test_part.sh: the shell side of tests/test_part.h. Source it (before any cd), then guard each independent section:
#   part 2 && { ...section 2... }
# UFSM_TEST_PART=K/N runs the sections whose index i has i % N == K; unset, every section runs. A section must build what it needs
# itself (a shard re-trains its own 3-step net: < 1 s on the fixture), so the shards of one test check exactly what the whole run does.
part() {
  [ -z "${UFSM_TEST_PART:-}" ] && return 0
  local k=${UFSM_TEST_PART%/*} n=${UFSM_TEST_PART#*/}
  [ "$n" -gt 1 ] 2> /dev/null || return 0
  [ $(( $1 % n )) -eq "$k" ]
}

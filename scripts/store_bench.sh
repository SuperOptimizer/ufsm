#!/usr/bin/env bash
# HTTP fetch A/B: runs build/store_bench (or $BENCH) once per configuration, rounds rotate the order (server-side caching
# favours later runs). Usage: store_bench.sh <root> <array-key> [rounds] [extra store-bench args...]
# CONFIGS (env) overrides the list: "name:VAR=v,VAR=v name2:..."
set -u
R=$1; K=$2; ROUNDS=${3:-1}; shift 3 2>/dev/null || shift $#; EXTRA=("$@")
BENCH=${BENCH:-$(dirname "$0")/../build/store_bench}
OLD="UFSM_HTTP_MULTI=0,UFSM_HTTP_TAIL=0,UFSM_Z3_INDEX_MEM=0,UFSM_HTTP_SHARE=0,UFSM_Z3_INDEX_PARALLEL=0"
CONFIGS=${CONFIGS:-"nocoal:$OLD,UFSM_Z3_COALESCE=0 master:$OLD idx:UFSM_HTTP_MULTI=0,UFSM_HTTP_SHARE=0 share:UFSM_HTTP_MULTI=0 multi:"}
for round in $(seq 1 "$ROUNDS"); do
  read -ra list <<< "$CONFIGS"; n=${#list[@]}
  for i in $(seq 0 $((n - 1))); do
    c=${list[$(( (i + round - 1) % n ))]}; name=${c%%:*}; vars=${c#*:}
    env $(echo "$vars" | tr ',' ' ') "$BENCH" "$R" "$K" --seed $((round * 17)) "${EXTRA[@]}" 2>/dev/null | sed "s/^/$name r$round /"
  done
done

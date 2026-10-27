#!/usr/bin/env bash
# join.sh RUN OUT FC Z0 [fit-join flags, e.g. --ref MODEL --chain-w 0]: fit-join of the slab fits RUN/sNNN/fit (cells files "c0 c1 a b")
# on the crop FC (its first field cell Z0), then the joined score -> OUT/join.log, OUT/score.log and a one-line summary.
# Env GPU (1), DETAIL ("--score-detail 1"; "" for whole-scroll scoring: 89 GB with detail).
set -u; . $(dirname $0)/env.sh; R=$1; O=$2; FC=$3; Z0=$4; shift 4; mkdir -p $O
args=(); for d in $(ls -d $R/s??? | sort); do [ -e $d/fit/model.json ] || continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * (c0 - Z0))) $((2 * (c1 - Z0)))); done
$U fit-join $O/joined $FC --blend 16 "$@" "${args[@]}" > $O/join.log 2>&1 || { echo "join failed $O"; tail -3 $O/join.log; exit 1; }
$U fit $FC --model $O/joined --gpu ${GPU:-1} --codes $Q 9.6 ${DETAIL---score-detail 1} --score-pairs 1 > $O/score.log 2>&1
echo "$O: $(grep -oE 'winding ids right [0-9.]+' $O/score.log | tail -1) | share $(grep -oE 'mean share\(mode\)[^:]*: [0-9.]+' $O/join.log | awk '{print $NF}') | agree $(grep -oE 'integer agreement[^:]*: [0-9.]+' $O/join.log | awk '{print $NF}') | refshare $(grep -oE "ref's integer[^:]*: [0-9.]+" $O/join.log | awk '{print $NF}') | pairs $(grep -oE '0.5 mm [0-9.]+|4 mm [0-9.]+' $O/score.log | head -2 | tr '\n' ' ')"

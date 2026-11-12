#!/bin/bash
# pod_bench.sh [REPS]: predict throughput on the pod, the 19.3 Gvox Paris 4 9.6 um box of DESIGN 2026-11-11 (the planner's window, --channel rv as univI's bench:
# one window per 2^3 shards), from the volcomp mirror (warm chunk cache) and from the latvol group $LV (ufsm latvol of the same level), alternating.
# Run under pod_claim.sh. Env: U (binary), CK (checkpoint), LV (latvol root), CACHE (volcomp chunk cache), TH (40), O (output dir), EXTRA.
U=${U:-/workspace/latstore/ufsm/build/ufsm}; CK=${CK:-/workspace/univI/u1b.ckpt}; LV=${LV:-/workspace/latstore/lv}; CACHE=${CACHE:-/workspace/univI/cache}
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; K=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
B=${B:-9216,2560,2560,2048,3072,3072}; O=${O:-/workspace/latstore/out}; TH=${TH:-40}; mkdir -p $O
one() {   # TAG ROOT [--cache DIR]
  local t=$1 r=$2; shift 2; rm -rf $O/$t
  env UFSM_PRED_PROF=1 UFSM_PRED_IO=1 $EXTRA $U predict $CK $r $K $O/$t --um 2.4 --level 2 --gpu 0 --threads $TH --box $B --channel rv "$@" > $O/$t.log 2>&1
  echo "$t: exit $? $(grep -E 'predict io: box' $O/$t.log | cut -c1-200)"; grep -E 'predict profile \(s\)|latvol CT' $O/$t.log | cut -c1-260 | sed "s/^/  /"
}
for i in $(seq ${1:-2}); do one vc$i $M --cache $CACHE; one lv$i $LV; done

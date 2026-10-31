#!/usr/bin/env bash
# slabs.sh RUN Z0 ZE GPU [fit flags]: the fine stage: production slab fits (64-cell interiors + 16 overlap, full cross-section of the
# union crop) over field cells [Z0, ZE) into RUN/sNNN (cells, fit.log, fit/), e.g. with --ref-model REF --uw-rep-ref 1. Claims by flock,
# so lanes on several GPUs / hosts (shared /vesuvius paths, or one RUN per host) fill one RUN. Env FIELDS (wind96), TH (12), NEED (GB
# available before a fit, 30), DMAX (field rows, 9473), UWR (--uw-repair tile, 8; 0 off), GLOCK (the GPU lock file, /tmp/ufsm-fit-gpuGPU.lock:
# start two lanes per GPU, each fit holds the lock from its device allocation to its exit, so one decodes fields while the other is on
# the GPU; "" none), CODES (1: score each slab against the Paris 4 codes; 0: no score, faster). Stop: touch RUN/STOP.
set -u; . $(dirname $0)/env.sh; R=$1; Z0=$2; ZE=$3; G=$4; shift 4; FI=${FIELDS:-$F96}; I=64; OV=16; DM=${DMAX:-9473}; mkdir -p $R
echo "$(date '+%m-%d %H:%M') $U $* fields $FI" >> $R/args.$HOST.$G
for c0 in $(seq $Z0 $I $((ZE - 1))); do
  k=$(( (c0 - Z0) / I )); s=$(printf 's%03d' $k); d=$R/$s; mkdir -p $d
  exec 9> $d/claim; flock -n 9 || continue; [ -e $d/done ] && continue
  [ -e $R/STOP ] && exit 0
  c1=$((c0 + I)); [ $c1 -gt $ZE ] && c1=$ZE; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $DM ] && b=$DM
  rm -rf $d/f $d/fit; bash $MK $FI $d/f $a $Y0 $X0 $((b - a)) $NY $NX > /dev/null; echo "$c0 $c1 $a $b" > $d/cells
  NEED=${NEED:-30} memok; t0=$(date +%s)
  env UFSM_FIT_THREADS=${TH:-12} UFSM_FIT_GPU_LOCK=${GLOCK-/tmp/ufsm-fit-gpu$G.lock} $U fit $d/f --gpu $G --out $d/fit $([ "${CODES:-1}" = 1 ] && echo --codes $Q 9.6) --dir 1 --uw-repair ${UWR:-8} "$@" > $d/fit.log 2>&1; rc=$?
  echo "$s $c0..$c1 rc $rc ids $(grep -oE 'winding ids right [0-9.]+' $d/fit.log | tail -1 | awk '{print $4}') $HOST gpu $G $(( $(date +%s) - t0 ))s" >> $R/slabs.txt
  [ $rc = 0 ] && touch $d/done
done

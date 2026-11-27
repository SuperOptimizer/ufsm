#!/bin/bash
# CPU sampler-only pass of a sources json (every device flag -1, CVD empty): N windows, strict sources; counts DEFAULTED. $1 sources, $2 tag, $3 N
set -u; cd /vesuvius/ufsm/atlas/ab
SRCJ=$1; T=$2; N=${3:-6}; OUT=/vesuvius/ufsm/atlas/ab/runs/cpu-$T; CTGD=-1; ONFLYGPU=-1; FIBGPU=-1; INKGPU=-1; GPUAUG=0; ECHOMAX=0; WORKERS=4; SLOTS=4; ONFLYMB=1024; P=${P:-384}
. ./args.sh; mkdir -p $OUT
CUDA_VISIBLE_DEVICES= UFSM_NCPU=6 UFSM_SAMPLER_THP=0 UFSM_SAMPLER_ONLY=$N UFSM_MESHCODES_CPU=1 timeout -k 10 ${TMO:-1800} ~/bin/memcap -m 16G ~/bin/lowprio $U train "${A[@]}" > cpu-$T.log 2>&1
rc=$?
echo "$(date +%T) $T rc=$rc DEFAULTED=$(grep -c DEFAULTED cpu-$T.log) $(grep -iE 'sampler-only|produced' cpu-$T.log | tail -1 | cut -c1-200)" >> cpu.log

#!/bin/bash
# pod_shared.sh CMD...: one unit of a shared pod GPU job per /workspace/ue/GPU.md: shared jobs run only in minutes :00-:20 of each hour
# (U2 has the card to itself otherwise), units <= 5 min, none started after :15 (SHSTART; a whole-crop ink predict: up to 8 min, started
# by :10). Waits while gpu.pause exists or, for my other units, while a sweep-B finetune trains (b.busy); marks gpu.shared/$SHN.
K=/workspace/ue; N=${SHN:-latstore}
while [ -e $K/gpu.pause ] || [ -e $K/gpu.gate-waiting ] || [ -e $K/gpu.merge-waiting ] || [ $((10#$(date +%M))) -ge ${SHSTART:-15} ] || { [ -e /workspace/latstore/b.busy ] && [ "${SHN:-}" != latstore-B ]; }; do sleep 5; done
mkdir -p $K/gpu.shared; touch $K/gpu.shared/$N; "$@"; rc=$?; rm -f $K/gpu.shared/$N; exit $rc

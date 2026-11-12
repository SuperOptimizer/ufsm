#!/bin/bash
# pod_shared.sh CMD...: one unit of a shared pod GPU job per /workspace/ue/GPU.md (waits while gpu.pause exists, marks gpu.shared/$SHN)
K=/workspace/ue; N=${SHN:-latstore}
while [ -e $K/gpu.pause ] || { [ -e /workspace/latstore/b.busy ] && [ "${SHN:-}" != latstore-B ]; }; do sleep 5; done   # (b.busy: a sweep-B finetune is training, my other units wait)
mkdir -p $K/gpu.shared; touch $K/gpu.shared/$N; "$@"; rc=$?; rm -f $K/gpu.shared/$N; exit $rc

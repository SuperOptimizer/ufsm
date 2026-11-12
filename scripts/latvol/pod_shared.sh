#!/bin/bash
# pod_shared.sh CMD...: one unit of a shared pod GPU job per /workspace/ue/GPU.md (waits while gpu.pause exists, marks gpu.shared/latstore)
K=/workspace/ue
while [ -e $K/gpu.pause ]; do sleep 5; done
mkdir -p $K/gpu.shared; touch $K/gpu.shared/latstore; "$@"; rc=$?; rm -f $K/gpu.shared/latstore; exit $rc

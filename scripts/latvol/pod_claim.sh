#!/bin/bash
# pod_claim.sh CMD...: CMD as an exclusive pod GPU job per /workspace/ue/GPU.md (claim gpu.pause by name, wait for gpu.shared/ empty, no
# eval.busy and no GPU process; run; release). Keep CMD under 15 min.
P=/workspace/ue/gpu.pause; me=latstore
while :; do
  while [ -e $P ] && [ "$(cat $P 2>/dev/null)" != $me ]; do sleep 2; done
  [ -e $P ] || echo $me > $P; sleep 1; [ "$(cat $P 2>/dev/null)" = $me ] && break
done
trap '[ "$(cat $P 2>/dev/null)" = '$me' ] && rm -f '$P EXIT
while [ -n "$(ls -A /workspace/ue/gpu.shared 2>/dev/null)" ] || [ -e /workspace/ue/eval.busy ] || nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 1; done
echo "claimed $(date +%T)"
"$@"

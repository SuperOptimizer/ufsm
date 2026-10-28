#!/bin/bash
# memguard: if MemAvailable < MIN GB, kill the corpus ink predicts (they resume from written shards) so forlindesk never thrashes again
MIN=${MIN:-8}
while true; do
  a=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
  if [ "$a" -lt "$MIN" ]; then
    for p in $(pgrep -f "ufsm predict .*ift3"); do echo "$(date "+%F %T") avail ${a}G: kill predict $p" >> /vesuvius/ufsm/corpus/memguard.log; kill $p; done
  fi
  sleep 15
done

#!/usr/bin/env bash
# scripts/univ/b1_pull_p4A.sh (desk): pulls the Paris 4 June 4.8 um phase shards from pod 2 (b1_pod_p4A.sh) as they are written, with
# rsync --remove-source-files (shards are written whole by rename; temporary names excluded), and marks each finished part as pulled so
# the pod starts the next. Ends when the pod log says done and nothing is left to pull.
set -u; EP=${EP:-root@213.173.107.10}; EPP=${EPP:-27064}; P="ssh -p $EPP -i $HOME/.ssh/id_ed25519"
R=${R:-/workspace/b1/gt/paris4/phaseA48}; L=${L:-/mnt/scratch/univ/b1/gt/paris4/phaseA48}; LOGF=${LOGF:-/workspace/b1/p4A.log}; mkdir -p $L
while :; do
  done_parts=$($P $EP "ls $R.part-*.done 2>/dev/null" < /dev/null)
  mkdir -p $L/4.8; rsync -a -e "$P" $EP:$R/zarr.json $L/zarr.json 2>/dev/null; rsync -a -e "$P" $EP:$R/4.8/zarr.json $L/4.8/zarr.json 2>/dev/null
  rsync -a --remove-source-files -e "$P" --exclude='*.tmp*' --exclude='.*' $EP:$R/4.8/c/ $L/4.8/c/ 2>/dev/null
  for f in $done_parts; do $P $EP "touch ${f%.done}.pulled" < /dev/null; done
  if $P $EP "test -e $R.all.done" < /dev/null; then
    left=$($P $EP "find $R/4.8/c -type f 2>/dev/null | wc -l" < /dev/null); [ "${left:-1}" = 0 ] && break
  fi
  sleep 60
done
touch $L/done; echo "$(date) pulled: $(find $L/4.8/c -type f | wc -l) shards, $(du -sh $L | cut -f1)"

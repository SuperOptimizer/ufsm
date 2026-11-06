#!/bin/bash
cd /mnt/scratch/v2/trace
echo "running: $(pgrep -fc 'corpus_trac[e]2')  done: $(tail -n +2 corpus/summary.tsv | wc -l)"
join -t $'\t' <(tail -n +2 corpus_v2/summary.tsv | cut -f1,6,8,9 | sort) <(tail -n +2 corpus/summary.tsv | cut -f1,6,8,9 | sort) |
  awk -F'\t' '{printf "%-12s kept %5s -> %5s  agree %5s -> %5s  viol %5s -> %5s\n", $1, $2, $5, $3, $6, $4, $7}'
echo -n "parallel: "; for s in $(tail -n +2 corpus/summary.tsv | cut -f1); do grep -q 'parallel rays' corpus/$s/trace.txt && echo -n "$s "; done; echo

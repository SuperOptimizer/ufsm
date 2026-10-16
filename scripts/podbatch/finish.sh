#!/usr/bin/env bash
# finish.sh: the end of fit-field batch v5-v7. Once every volume (queue.list + PHerc1667 + PHerc0009B) is uploaded or failed and no
# gpu_lane / adopt / up_one / lftp / predict process is left: if every predicted volume is uploaded, the upload credentials
# (/root/.ash2txt) are shredded; otherwise they are kept and the missing volumes logged. out/FINAL.md: the per-volume STATUS lines.
set -u; cd /workspace; ST=out/STATUS.md
log() { ( flock 8; echo "$(date -u +%Y-%m-%dT%H:%MZ) finish $*" >> $ST ) 8> status.lock; }
log "watching (pid $$)"
while :; do
  left=$( { cut -d" " -f1 queue.list; echo PHerc1667; echo PHerc0009B; } | while read -r s; do [ -e state/$s.uploaded ] || [ -e state/$s.failed ] || echo $s; done | wc -l)
  busy=$(ps -eo args | grep -c -E "^(bash \./gpu_lane|bash \./adopt|bash \./up_one|lftp -f|/workspace/ufsm/build/ufsm predict)")
  [ $left = 0 ] && [ $busy = 0 ] && break
  if [ $busy = 0 ] && ! ps -eo args | grep -q -E "^bash \./pf_lane"; then
    log "nothing running but $left volume(s) neither uploaded nor failed: $( { cut -d" " -f1 queue.list; echo PHerc1667; echo PHerc0009B; } | while read -r s; do [ -e state/$s.uploaded ] || [ -e state/$s.failed ] || echo -n "$s "; done); credentials kept"; exit 1; fi
  sleep 120
done
grep -E "predicted in|uploaded [0-9]+ files|FAILED|giving up" $ST > out/FINAL.md
miss=$(for f in state/*.predicted; do s=$(basename $f .predicted); [ -e state/$s.uploaded ] || echo -n "$s "; done)
nf=$(ls state/*.failed 2>/dev/null | wc -l)
if [ -z "$miss" ]; then shred -u /root/.ash2txt && log "all uploads done ($nf failed volume(s)); /root/.ash2txt shredded"
else log "predicted but not uploaded: $miss; /root/.ash2txt kept"; fi

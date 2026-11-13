#!/bin/bash
# podpull.sh (desk; env PODH=root@HOST PODPORT PODKEY): a pod's unroll queue (scripts/pod/q.sh) -> desk, as each scroll's stripimg lands (user: "send scroll renders as they land"): renders first
# (overview.jpg, strip_long.jpg, one ct + ink stripe; temp name + rename), then the small rows (progress.log, wrapcheck.tsv, state, tsvs);
# never wraps / fields. One line per event -> sched/podq.log
H=${PODH:?PODH=root@HOST}; SSH="ssh -p ${PODPORT:-22} -i ${PODKEY:-$HOME/.ssh/id_ed25519} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
SD=/vesuvius/ufsm/unroll/sched; L=$SD/podq.log; RU=/root/pod/unroll
get() { local src=$1 dst=$2; mkdir -p "$(dirname $dst)"; rsync -a -e "$SSH" "$H:$src" "$dst.part" 2>/dev/null && mv -f "$dst.part" "$dst"; }
while :; do
  for S in $($SSH $H 'cat /root/pod/kit/q.txt' 2>/dev/null); do
    D=/vesuvius/ufsm/unroll/$S-u1; [ -e $D/.pulled ] && continue
    st=$($SSH $H "ls $RU/$S/state 2>/dev/null | tr '\n' ' '; ls /root/pod/qstate/$S.fail 2>/dev/null" 2>/dev/null)
    case "$st" in *stripimg.done*) ;; *"$S.fail"*) [ -e $D/.failnoted ] || { mkdir -p $D; touch $D/.failnoted; echo "$(date '+%F %T') $S FAILED on the pod: $($SSH $H "tail -2 /root/pod/unroll/$S/progress.log" | tr '\n' ' ' | cut -c1-300)" >> $L; get $RU/$S/progress.log $D/progress.log; }; continue;; esac
    case "$st" in *stripimg.done*) ;; *) continue;; esac   # (a rerun that lands after a failure is pulled)
    I=$D/strip/img_fill; R=$RU/$S/strip/img_fill
    ws=$($SSH $H "cd $R && ls ws*_ct.jpg | sort | awk '{a[NR]=\$0} END {print a[int((NR + 1) / 2)]}'")
    for f in strip_long.jpg ${ws} ${ws/_ct/_ink} papyrus_wraps.tsv strip_papyrus.tsv stretch.txt; do get $R/$f $I/$f; done
    get $R/overview.jpg $I/overview.jpg
    get $RU/$S/progress.log $D/progress.log; get $RU/$S/wrapcheck.tsv $D/wrapcheck.tsv; get $RU/$S/slabs.progress $D/slabs.progress
    get $RU/$S/box.txt $D/box.txt; get $RU/$S/axis.json $D/axis.json; rsync -a -e "$SSH" $H:$RU/$S/state/ $D/state/ 2>/dev/null
    m=$(grep -h "strip-mosaic:" $D/progress.log | tail -1 | sed -E 's/.*strip-mosaic: ([0-9]+ stripes), length ([0-9.]+ cm), valid ([0-9.]+ cm2), on papyrus[^0-9]*\) ([0-9.]+).*/\1, \2, \3, pap \4/')
    w=$(grep -h "wrap-check: verdicts" $D/progress.log | tail -1 | sed 's/.*verdicts: //')
    t=$(grep -h "finished:" $D/progress.log | tail -1 | grep -oE "fields [0-9]+ s wall|slabs [0-9]+ s wall" | tr '\n' ' ')
    # sanity gate (lead 11-12, after MANB): on-papyrus < 0.5 or dup share > 30 % of the wrap-check verdicts -> FLAG
    pap=$(grep -h "strip-mosaic:" $D/progress.log | tail -1 | grep -oE "on papyrus \(CT[^)]*\) [0-9.]+" | awk '{print $NF}')
    fl=$(echo "$w" | awk -v p="${pap:-0}" '{r = $1; d = $3; m = $5; n = r + d + m; s = n ? d / n : 0; f = ""; if (p < 0.5) f = f sprintf(" on-papyrus %.3f < 0.5", p); if (s > 0.3) f = f sprintf(" dup share %.0f%% > 30%%", 100 * s); print f}')
    touch $D/.pulled; echo "$(date '+%F %T') $S renders landed: $I/overview.jpg ($m); wrap-check $w; $t${fl:+ ** FLAG:$fl}" >> $L
    # auto-trim on the pod once the desk has the images (lead 11-13, pod / full at 00:24): the strip mesh, ink store, count-field ref and
    # the CT copy go (CT: refetched by setup.sh if the scroll is rerun, so its state/ct markers go too); joined, wraps, wraps_snap,
    # img_fill / img_pap, logs and state stay for QA
    if [ -s $I/overview.jpg ] && [[ $S == PHerc* ]]; then
      fr=$($SSH $H "rm -rf /root/pod/unroll/$S/strip/strip /root/pod/unroll/$S/ink /root/pod/unroll/$S/ref-cf /root/pod/vol/$S /root/pod/state/ct.$S.*; df -BG --output=avail / | tail -1" 2>/dev/null | tr -dc 0-9)
      echo "$(date '+%F %T') $S trimmed on the pod: ${fr:-?} GB free" >> $L; fi
  done
  sleep 120
done

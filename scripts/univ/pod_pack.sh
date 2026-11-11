#!/usr/bin/env bash
# scripts/univ/pod_pack.sh OUT [SNAP]: on the desk, stage everything a fresh universal-net pod cannot fetch itself (scripts/univ/pod_univ.sh),
# laid out as the pod's K: OUT/small (the datakit layout: axis / umbilicus JSONs, gtaxis, klt1.lcb, holdouts, boxes, wind96, the init
# checkpoints) and OUT/gt (the label inputs with no public source: PHerc0139 q-surfaces 55 MB, PHerc1667 codes A / B 9.4 GB + its 9.6 um
# phase 9.8 GB). SNAP: a u1 snapshot (or last.ckpt) to resume on the pod, copied with its run's ufsm.bin (reference only:
# the desk binary needs glibc 2.43; pod_univ.sh setup rebuilds COMMIT from source) and COMMIT noted. Also OUT/md5: md5
# sums of 8 sampled label shards per set, which pod_univ.sh `verify` checks against its rebuilt rasters. Everything else (CT, segments,
# Paris 4 / 0139 / 0500P2 labels) the pod reads from the mirror or rebuilds (datakit.sh). Then from the desk:
#   rsync -a --info=progress2 -e "ssh -p PORT -i ~/.ssh/id_ed25519" OUT/ root@HOST:/workspace/univ/      (~20 GB, ~75 min at 4.5 MB/s)
# Hard links where possible (OUT on the same filesystem as the sources), so staging costs no disk.
set -euo pipefail; O=${1:?OUT}; SNAP=${2:-}
mkdir -p $O/small/axis $O/small/gtaxis $O/gt/p0139-q $O/gt/1667 $O/md5 $O/ckpt
cpl() { cp -al "$1" "$2" 2>/dev/null || cp -a "$1" "$2"; }   # (hard link, else copy)
cpl /vesuvius/ufsm/teacher/axis-0.json $O/small/axis/paris4-axis-0.json
cpl /vesuvius/ufsm/umbilicus/PHerc0139/umbilicus-full-resolution.json $O/small/axis/PHerc0139-umbilicus.json
cpl /mnt/scratch/pods/PHerc0139/unroll/axis.json $O/small/axis/PHerc0139-unroll-axis.json
cpl /vesuvius/ufsm/v2/rv/gt/axis/PHerc0500P2-20250820143440.json $O/small/axis/PHerc0500P2-20250820143440.json
cpl /vesuvius/ufsm/umbilicus/PHerc1667/umbilicus-full-resolution.json $O/small/axis/PHerc1667-umbilicus.json
cpl /mnt/scratch/pods/PHerc1667/unroll/axis.json $O/small/axis/PHerc1667-unroll-axis.json
for f in /vesuvius/ufsm/v2/rv/gt/axis/*.json; do cpl $f $O/small/gtaxis/; done
cpl /vesuvius/ufsm/v2/lat/codec/klt1.lcb $O/small/klt1.lcb
cpl /vesuvius/ufsm/wbench/holdouts.json $O/small/holdouts.json; cpl /vesuvius/ufsm/wbench/boxes.tsv $O/small/boxes.tsv
cpl /vesuvius/ufsm/podready/pod/ckpt/wind96-wk1-best.ckpt $O/small/wind96-wk1-best.ckpt
cpl /vesuvius/ufsm/v2/lat/runs/wz896/last.ckpt $O/ckpt/wz896.ckpt
cpl /vesuvius/ufsm/univ/u1/sources-u1.json $O/small/sources-u1-desk.json
G=/vesuvius/ufsm/gt/PHerc0139-wind-20261016; [ -d $O/gt/p0139-q/qA ] || { cp -al $G/qA $G/qB $O/gt/p0139-q/ 2>/dev/null || cp -a $G/qA $G/qB $O/gt/p0139-q/; }
G=/vesuvius/ufsm/gt/PHerc1667-wind-20261016; for d in codesA codesB phase; do [ -d $O/gt/1667/$d ] || { cp -al $G/$d $O/gt/1667/ 2>/dev/null || cp -a $G/$d $O/gt/1667/; }; done
# md5 samples of the label sets the pod rebuilds (8 shards each, fixed by sort order; shards without labels are absent on both sides)
smp() { local n=$1 d=$2; (cd $d && find . -path '*/c/*' -type f | sort | awk 'NR % 97 == 1' | head -8 | xargs -r md5sum) > $O/md5/$n.md5; echo "$n: $(wc -l < $O/md5/$n.md5) shards"; }
smp p4A /vesuvius/ufsm/v2/rv/gt/p4full/phaseA; smp p4B /vesuvius/ufsm/v2/rv/gt/p4full/phaseB; smp p0139 /vesuvius/ufsm/v2/rv/gt/p0139full/phase
smp p0500 /vesuvius/ufsm/v2/rv/gt/p0500p2full/phase; smp 1667A48 /mnt/scratch/univ/b1/gt/1667/phaseA48; smp 1667B48 /mnt/scratch/univ/b1/gt/1667/phaseB48
if [ -n "$SNAP" ]; then   # the run to resume: its checkpoint, binary and the commit that built it
  cpl $SNAP $O/ckpt/u1-resume.ckpt; R=$(dirname $SNAP); [ -f $R/ufsm.bin ] || R=$(dirname $R)
  cpl $R/ufsm.bin $O/ckpt/u1-ufsm.bin; echo ${COMMIT:?COMMIT: the commit that built the run ufsm.bin, u1: 23c085b} > $O/ckpt/u1-commit.txt
  head -1 $SNAP | grep -oE '"step":[0-9]+' > $O/ckpt/u1-step.txt || true
fi
du -shL $O; echo "staged: rsync -a --info=progress2 -e 'ssh -p PORT -i ~/.ssh/id_ed25519' $O/ root@HOST:/workspace/univ/"

#!/bin/bash
# scripts/atlas/gate.sh: the ATLAS-15k gate as a filter (2026-11-27, atlas). Kept patches are written with a q.tif (winding in our fit's gauge)
# as `meshes` onfly sets: $K/<scroll>/<id>.tifxyz. Counts per scroll and stage -> $K/gate.tsv.
#   Stages (a patch leaves at the first it fails):
#   excluded scroll  0841 / 1447 (ink holdouts), 0139 (the rv eval's p139 scroll; its GT calibrates the gate), 0814 (GT too poor, no wraps),
#                    scrolls without wraps of ours (1218 0846B 0268 0490B: nothing to check switches against)
#   held             the patch bbox + 16 voxels touches a box of $HO (every wbench box, holdouts-20261114, J1, the ATLAS eval slabs)
#   align            CT lamination share < 0.6 (wrap-align, audit/align.tsv)
#   nocover / fail   vs our wraps: matched < 0.3; le4 < 0.7 (not on a sheet our fit sees); big < 0.271 (sheet switches: worse than
#                    10 % of HercUeNet's approved 1203 sheet against the same wraps); amb > 0.052 (merge: a second wrap within 4 voxels on
#                    more on-sheet nodes than HercUeNet's p90)
# Env: U (ufsm), A (/vesuvius/ufsm/atlas), K (kept dir), HO (holdouts json), NCPU (6)
set -u
U=${U:-/vesuvius/ufsm/bin/ufsm-atlas}; A=${A:-/vesuvius/ufsm/atlas}; K=${K:-$A/kept}; HO=${HO:-$A/holdouts-atlas.json}; NCPU=${NCPU:-6}
M=/vesuvius/ufsm
mkdir -p $K
awk -F'\t' '$3 >= 0.6 { print $1 }' $A/audit/align.tsv > $K/allow.txt
echo -e "scroll\tpatches\tpass\tfail\tnocover\theld\tdropped_align\twritten" > $K/gate.tsv.tmp
while read s w ax; do
  [ -d $K/$s ] && [ -e $K/$s.done ] && { grep -P "^$s\t" $K/gate.tsv >> $K/gate.tsv.tmp; continue; }
  mkdir -p $K/$s
  log=$(UFSM_NCPU=$NCPU ~/bin/memcap -m 16G ~/bin/lowprio $U atlas-audit $A/surf/index.tsv --scroll $s --ref $M/$w --holdouts $HO --allow $K/allow.txt \
        --gate-d 99 --gate-le 0.7 --gate-big 0.271 --gate-amb 0.052 --gate-sw 1 --write-q $K/$s --axis $ax --out $K/$s.audit.tsv 2>&1 | tail -1)
  echo "$s: $log"
  n=$(echo "$log" | grep -o '^atlas-audit: [0-9]*' | grep -o '[0-9]*$')
  echo "$log" | awk -v s=$s -v n=$n '{ for (i = 1; i <= NF; i++) { if ($i == "pass") p = $(i+1); if ($i == "fail") f = $(i+1); if ($i == "nocover") c = $(i+1); if ($i == "box)") h = $(i+1); if ($i == "(--allow)") d = $(i+1); if ($i == "written") w = $(i-1) }
     gsub(/[,;]/, "", p); gsub(/[,;]/, "", f); gsub(/[,;]/, "", c); gsub(/[,;]/, "", h); gsub(/[,;]/, "", d); printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", s, n, p, f, c, h, d, w }' >> $K/gate.tsv.tmp
  touch $K/$s.done
done <<EOF
PHerc0800 scratch-mig/pods/PHerc0800/unroll/wraps $M/pods/PHerc0800/unroll/axis.json
PHerc0306B scratch-mig/pods/PHerc0306B/run/wraps $M/scratch-mig/pods/PHerc0306B/run/axis.json
PHerc0175A scratch-mig/pods/PHerc0175A/unroll_v2/wraps $M/scratch-mig/pods/PHerc0175A/unroll/axis.json
PHerc1203 unrollrun/PHerc1203-p/wraps $M/unrollrun/PHerc1203-p/axis.json
PHerc0125 pods/PHerc0125/run/wraps $M/pods/PHerc0125/run/axis.json
PHerc1545 scratch-mig/pods/PHerc1545/unroll/wraps $M/scratch-mig/pods/PHerc1545/unroll/axis.json
PHerc0826 scratch-mig/pods/PHerc0826/unroll/wraps $M/scratch-mig/pods/PHerc0826/unroll/axis.json
PHerc0813 pods/PHerc0813/p0813_cal/run/wraps $M/scratch-mig/pods/PHerc0813/p0813/axis.json
PHerc0846A scratch-mig/pods/PHerc0846A/run/wraps $M/scratch-mig/pods/PHerc0846A/run/axis.json
PHerc0009B scratch-mig/pods/PHerc0009B/run/wraps $M/scratch-mig/pods/PHerc0009B/run/axis.json
PHerc0175B pods/PHerc0175B/run_v2/wraps $M/scratch-mig/pods/PHerc0175B/run_v2/axis.json
PHerc0191 pods/PHerc0191/run/wraps $M/pods/PHerc0191/run/axis.json
PHerc0211 pods/PHerc0211/unroll_v2/wraps $M/pods/PHerc0211/unroll/axis.json
PHerc0257 pods/PHerc0257/unroll/wraps $M/pods/PHerc0257/unroll/axis.json
PHerc0343 pods/PHerc0343/unroll/wraps $M/scratch-mig/pods/PHerc0343/unroll/axis.json
PHerc0358 scratch-mig/pods/PHerc0358/run/wraps $M/scratch-mig/pods/PHerc0358/run/axis.json
PHerc0483A scratch-mig/pods/PHerc0483A/run_v2/wraps $M/pods/PHerc0483A/run/axis.json
PHerc0483B scratch-mig/pods/PHerc0483B/run_v2/wraps $M/scratch-mig/pods/PHerc0483B/run_v2/axis.json
PHerc0490A scratch-mig/pods/PHerc0490A/unroll/wraps $M/scratch-mig/pods/PHerc0490A/unroll/axis.json
PHerc1451 pods/PHerc1451/unroll/wraps $M/scratch-mig/pods/PHerc1451/unroll/axis.json
EOF
mv $K/gate.tsv.tmp $K/gate.tsv
echo DONE

#!/bin/bash
# corpus_trace.sh [SCROLL..]: sheet-trace on one 512^3 sample box per corpus scroll (GT-free QA). Box: mid axis z, centred at FRAC (0.5)
# of the way from the axis to the volume edge along +x. Inputs: scrolls.tsv (axis, ct_key, ct_level), mirror m7 prob of the same scan.
# Out: /mnt/scratch/v2/trace/corpus/<scroll>/{trace,solve}.txt and one summary line per scroll in corpus/summary.tsv
set -uo pipefail
# env: STX = extra sheet-trace args, OUT = output dir (default corpus)
U=${U:-/vesuvius/ufsm/v3/ps-src/build/ufsm}; T=/mnt/scratch/v2/trace/${OUT:-corpus}; TSV=/vesuvius/ufsm/v2/scrolls.tsv; FRAC=${FRAC:-0.5}
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; C=/mnt/scratch/vesuvius-offload/cache
mkdir -p $T
[ -s $T/summary.tsv ] || printf "scroll\tum\tbox\tcols\tpeaks_per_col\tkept_share\ttiles\tpairs_agree\tviolated\tactive\tsecs\n" > $T/summary.tsv
awk -F'\t' 'NR>1 {print $1, $8, $10, $11, $12}' $TSV | while read -r S AX CTK UM LV; do
  [ $# -gt 0 ] && [[ " $* " != *" $S "* ]] && continue
  D=$T/$S; mkdir -p $D
  scan=$(basename "$CTK" | cut -d- -f1)
  pz=$(curl -s "$M/$S/representations/predictions/surfaces/" | grep -o "href=\"$scan-surface-[^\"]*-m7-L[0-9]-prob.zarr/\"" | head -1 | sed 's/href="//; s|/"||')
  if [ -z "$pz" ]; then echo "$S: no m7 prob for scan $scan" | tee $D/err.txt; continue; fi
  info=$($U info "$M/$S/representations/predictions/surfaces" "$pz" 2>/dev/null | sed -n 2p)
  key=$(echo "$info" | awk '{print $1}'); um=$(echo "$info" | awk '{print $2}')
  read -r NZ NY NX <<< "$(echo "$info" | sed -E 's/.*shape ([0-9]+) x ([0-9]+) x ([0-9]+).*/\1 \2 \3/')"
  # axis voxels -> prob grid: the axis is in CT level-LV voxels of ct_um; the prob grid is um
  AXS=$(echo "$UM / $um" | bc -l)   # the axis is in CT level-0 voxels (2.4 um scans too)
  read -r az ay ax <<< "$($U axis-at "$AX" 2>/dev/null)"
  if [ -z "${az:-}" ]; then   # mid control point (python-free: awk over the json)
    read -r az ay ax <<< "$(tr '{' '\n' < "$AX" | grep -o '"[xyz]": *-\?[0-9.e+-]*' | tr -d '" ' | awk -F: '{v[$1]=$2; if ($1=="x") {n++; X[n]=v["x"]; Y[n]=v["y"]; Z[n]=v["z"]}} END {m=int((n+1)/2); print Z[m], Y[m], X[m]}')"
  fi
  zc=$(echo "$az * $AXS" | bc -l | cut -d. -f1); yc=$(echo "$ay * $AXS" | bc -l | cut -d. -f1); xa=$(echo "$ax * $AXS" | bc -l | cut -d. -f1)
  # papyrus extent along +x / -x from the axis on the level-3 prob (8x coarser): last voxel >= 40; box centre FRAC of the way out
  k3=$($U info "$M/$S/representations/predictions/surfaces" "$pz" 2>/dev/null | sed -n 5p | awk '{print $1}')
  W3=$((NX / 8 + 1)); $U read "$M/$S/representations/predictions/surfaces" "$pz/$k3" $((zc / 8)) $((yc / 8)) 0 1 1 $W3 $D/row.raw --cache $C > /dev/null 2>&1
  read -r lo hi <<< "$(od -An -tu1 -v $D/row.raw | tr -s ' ' '\n' | awk -v a=$((xa / 8)) 'NF { v[n++] = $1 } END { lo = a; hi = a; for (i = 0; i < n; i++) if (v[i] >= 40) { if (i < lo) lo = i; if (i > hi) hi = i } print lo, hi }')"
  if [ $(( hi - xa / 8 )) -ge $(( xa / 8 - lo )) ]; then xc=$(( xa + $(echo "($hi * 8 - $xa) * $FRAC" | bc -l | cut -d. -f1) ));
  else xc=$(( xa - $(echo "($xa - $lo * 8) * $FRAC" | bc -l | cut -d. -f1) )); fi
  z0=$((zc - 256)); y0=$((yc - 256)); x0=$((xc - 256)); BOX=$z0,$y0,$x0,512,512,512
  t0=$(date +%s)
  rm -rf $D/st; $U sheet-trace --prob "$M/$S/representations/predictions/surfaces" --prob-key "$pz/$key" --cache $C --box $BOX --axis "$AX" --axis-scale $AXS \
      --out $D/st --vs $um --tile 16 --threads 16 --jpg $D/qa.jpg ${STX:-} > $D/trace.txt 2>&1
  $U patch-solve --paths $D/st/paths --prob "$M/$S/representations/predictions/surfaces" --prob-key "$pz/$key" --cache $C --box $BOX --axis "$AX" --axis-scale $AXS \
      --threads 16 --cut 0 --minv 32 --links $D/st/links.tsv --frame $D/st/frame.txt --tsv $D/nodes.tsv > $D/solve.txt 2>&1
  t1=$(date +%s)
  cols=$(grep -o 'columns [0-9]* x [0-9]*' $D/trace.txt | head -1 | sed 's/columns //; s/ //g'); ppc=$(grep -o '([0-9.]* per column)' $D/trace.txt | tr -d '(' | cut -d' ' -f1)
  kept=$(grep -o '= [0-9.]* of the peaks' $D/trace.txt | cut -d' ' -f2); tiles=$(grep -o 'as [0-9]* tiles' $D/trace.txt | cut -d' ' -f2)
  agree=$(grep -o 'mean vote agreement [0-9.]*' $D/solve.txt | cut -d' ' -f4)
  last=$(grep '^round' $D/solve.txt | tail -1); viol=$(echo "$last" | grep -o 'share) [0-9.]*' | cut -d' ' -f2); act=$(echo "$last" | grep -o 'active [0-9]*' | cut -d' ' -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $S $um $BOX "$cols" "$ppc" "$kept" "$tiles" "$agree" "$viol" "$act" $((t1 - t0)) | tee -a $T/summary.tsv
done

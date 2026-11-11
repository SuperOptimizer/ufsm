#!/bin/bash
# batchh2.sh: crops on human-labelled sheets (hcrops.txt) at 2.4 um and 1.129 um with the human labels, AUC vs HUMAN labels, contact sheets
cd /vesuvius/ufsm/pbr; U=/vesuvius/ufsm/pbr/ufsm.pbr4; C="--cache /vesuvius/ufsm/cache --threads 8 --step 0.25 --labels /vesuvius/ufsm/pbr/hlab 2.4"
R="--ct https://dl.ash2txt.org/community-uploads/forrest/volcomp PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"
while read nm seg u0 v0 n; do
  python3 -I inkcrop.py seg/$seg $v0 $u0 $n out/pred_${nm}.png
  for kind in s24 s113; do o=out/h_${kind}_$nm; echo "== $o"
    if [ $kind = s113 ]; then ~/bin/memcap -m 24G nice $U sheet-pbr seg/$seg/mesh/tifxyz $o --reg reg113/reg.json --crop $u0,$v0,$n --ss 2 --depth 85 $C 2>&1 | grep -E "fine CT|outside|labels:|DP|samples" | cut -c1-220
    else ~/bin/memcap -m 24G nice $U sheet-pbr seg/$seg/mesh/tifxyz $o $R --crop $u0,$v0,$n --depth 40 $C 2>&1 | grep -E "labels:|DP|samples" | cut -c1-220; fi
    [ -e $o/map.json ] || continue
    nice python3 -I auc.py $o human 300 2>&1 | grep -v Warning > $o/auc_human.txt; head -2 $o/auc_human.txt
    python3 -I sheeth.py $o "Paris 4 $nm ($seg, u $u0 v $v0, $n px of 2.4 um) - $kind" /vesuvius/ufsm/pbr/out/pred_${nm}.png > /dev/null
  done
done < hcrops.txt
echo batchh2 done

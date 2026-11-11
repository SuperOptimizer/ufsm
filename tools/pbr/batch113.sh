#!/bin/bash
# batch113.sh: 1024 x 1024 px sub-crops rendered from the 2.4 um scan (ss 1 and ss 2) and from the registered 1.129 um scan (ss 2: the
# 1.129 um pitch along the surface; offsets in 1.129 um voxels, depth 85 = 40 x 2.126), AUC vs the published ink prediction, contact sheets
cd /vesuvius/ufsm/pbr
while read nm seg u0 v0; do
  echo "== $nm $seg $u0 $v0"
  python3 -I inkcrop.py seg/$seg $v0 $u0 1024 out/sub_${nm}_ink.png
  [ -e out/s24_$nm/map.json ] || ./run.sh $seg $u0 $v0 1024 out/s24_$nm --depth 40 --step 0.25 2>&1 | grep -E "DP|samples" | cut -c1-200
  [ -e out/s24x2_$nm/map.json ] || ./run.sh $seg $u0 $v0 1024 out/s24x2_$nm --depth 40 --step 0.25 --ss 2 2>&1 | grep -E "DP|samples" | cut -c1-200
  [ -e out/s113_$nm/map.json ] || ~/bin/memcap -m 24G nice ./ufsm.pbr3 sheet-pbr seg/$seg/mesh/tifxyz out/s113_$nm --reg reg113/reg.json --cache /vesuvius/ufsm/cache \
      --crop $u0,$v0,1024 --ss 2 --depth 85 --step 0.25 --threads 8 2>&1 | grep -E "fine CT|DP|samples" | cut -c1-200
  for d in s24 s24x2 s113; do nice python3 -I auc.py out/${d}_$nm out/sub_${nm}_ink.npy 300 2>&1 | grep -v Warning > out/${d}_$nm/auc300.txt; python3 -I sheet.py out/${d}_$nm sub_$nm > /dev/null; done
done <<'L'
w046 20260701183127-w046-052 68096 38400
w028s 20260701183125-w028-037 103424 27648
w038s 20260701183126-w038-045 90112 44032
L
echo batch113 done

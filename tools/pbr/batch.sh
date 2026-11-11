#!/bin/bash
# batch.sh: the three crops (crops.json), sheet-pbr + AUC + contact sheet each
cd /vesuvius/ufsm/pbr
for s in w046-052 w038-045 w028-037; do full=$(ls seg | grep -- "-$s"); read u v < <(python3 -I -c "import json; y,x=json.load(open('crops.json'))['$full']; print(x,y)")
  echo "== $s"; ./run.sh $full $u $v 2048 out/$s --depth 40 --step 0.25 2>&1 | grep -E "DP|faces:|samples" | cut -c1-300
  nice python3 -I auc.py out/$s out/${s}_ink.npy 300 2>&1 | grep -v Warning > out/$s/auc300.txt; python3 -I sheet.py out/$s $s
done; echo batch done

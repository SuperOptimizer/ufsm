#!/bin/bash
# scripts/inkgt/splat_eval.sh: the held-out ink eval crops (wbench/inkeval/crops: 16 Paris 4; inkgt/evcrops: 8 PHerc0139 / 1667) splatted the same
# way as the training targets (--face -6,0, 4.8 um, iso 64) into $O/<crop>.zarr: the 3-D reference of each held-out region (voxelwise AUC).
set -u
U=${U:-/home/forrest/ufsm-inkgt/build/ufsm}; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; O=${O:-/vesuvius/ufsm/inkgt/evgt}; mkdir -p $O
P4K=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
for c in /vesuvius/ufsm/wbench/inkeval/crops/w*_? /vesuvius/ufsm/inkgt/evcrops/PHerc*; do
  n=$(basename $c); [ -s $O/$n.zarr/zarr.json ] && continue
  read key um < <(python3 -I -c "import json,sys; j=json.load(open(sys.argv[1])); print(j.get('ct','$P4K'), j.get('um', 2.4))" $c/crop.json)
  ~/bin/memcap -m 8G nice -n 10 $U ink-splat $O/$n.zarr --surface $c --labels $c/labels.tif --mask $c/mask.tif --ct $M $key --ct-um $um --level 1 --iso 64 \
    --face -6,0 --coarse 0 --name $n --threads 4 --cache /vesuvius/ufsm/cache > $O/$n.log 2>&1 || echo "FAIL $n"
  echo "$n $(grep -o 'open face found at [0-9]* of [0-9]* supervised px ([0-9.]*%)' $O/$n.log) $(grep -o '[0-9]* ink + [0-9]* no-ink written' $O/$n.log)"
done

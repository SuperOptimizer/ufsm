#!/bin/bash
# trace_box.sh NAME Z Y X NZ NY NX [NSEED 64] [PAR 6]: VC3D patches on the box radial ridge (NAME/ridge.raw from boxeval.sh) + normal grids, scored vs GT
set -e; N=$1; Z=$2; Y=$3; X=$4; NZ=$5; NY=$6; NX=$7; NS=${8:-64}; PAR=${9:-6}
T=/mnt/scratch/v2/trace; D=$T/$N; IMG=localhost/vc-local:20260828; cd $D
~/miniconda3/bin/python3 - <<PY
import numpy as np, zarr, json
a=np.fromfile("ridge.raw",np.uint8).reshape($NZ,$NY,$NX); a=np.where(a>0,255,0).astype(np.uint8)
g=zarr.open_group("ridge.zarr",mode="w",zarr_format=2); cur=a
for l in range(4):
    g.create_array(str(l),shape=cur.shape,dtype="u1",chunks=(64,64,64),fill_value=0); g[str(l)][:]=cur
    z,y,x=[s//2*2 for s in cur.shape]; cur=cur[:z,:y,:x].reshape(z//2,2,y//2,2,x//2,2).max(axis=(1,3,5))
g.attrs["multiscales"]=[{"version":"0.4","name":"$N","axes":[{"name":c,"type":"space","unit":"micrometer"} for c in "zyx"],"datasets":[{"path":str(l),"coordinateTransformations":[{"type":"scale","scale":[$VS*2**l]*3}]} for l in range(4)]}]
json.dump({"type":"vol","uuid":"$N","name":"$N","width":$NX,"height":$NY,"slices":$NZ,"voxelsize":$VS,"min":0.0,"max":255.0,"format":"zarr"},open("ridge.zarr/meta.json","w"))
z,y,x=np.nonzero(a[64:$NZ-64:32,64:$NY-64,64:$NX-64]>0); i=np.random.default_rng(2).choice(len(z),$NS,replace=False)
open("seeds.txt","w").write("".join("%d %d %d\n"%(64+x[k],64+y[k],64+32*z[k]) for k in i))
json.dump({"mode":"seed","step_size":5,"generations":400,"min_area_cm":0.005,"thread_limit":2,"voxelsize":$VS,"cache_root":"$D/vccache","tgt_overlap_count":10,"search_effort":10,"normal_grid_path":"$D/ngrid"},open("seed.json","w"))
PY
rm -rf ngrid vccache paths logs; mkdir -p vccache paths logs
podman run --rm -v $T:$T $IMG vc_gen_normalgrids -i $D/ridge.zarr -o $D/ngrid --spiral-step 5 --grid-step 32 --preview-every 0 > ngrid.log 2>&1
i=0
while read x y z; do
  ( timeout 1800 podman run --rm --cpus 2 -v $T:$T $IMG vc_grow_seg_from_seed -v $D/ridge.zarr -t $D/paths -p $D/seed.json -s $x $y $z > logs/s$i.log 2>&1 ) &
  i=$((i+1)); while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 2; done
done < seeds.txt
wait
~/miniconda3/bin/python3 $T/score_patches.py paths . $Z $Y $X $NZ $NY $NX $AXIS 0.25 $SGN | tee score.txt

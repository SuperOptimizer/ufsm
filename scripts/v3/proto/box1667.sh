#!/bin/bash
# box1667.sh NAME Z Y X NZ NY NX: PHerc1667 box (9.596 um): GT codes, m7 prob -> radial ridge (ridge.raw), detector scores; then trace_box-style VC3D patches + score
set -e; N=$1; Z=$2; Y=$3; X=$4; NZ=$5; NY=$6; NX=$7; U=/vesuvius/ufsm/v2/strip-src/build/ufsm; T=/mnt/scratch/v2/trace; D=$T/$N
mkdir -p $D; cd $D
$U read /mnt/scratch/v2/anchor/gt1667/codes.zarr 9.596 $Z $Y $X $NZ $NY $NX gt.raw --threads 8 > /dev/null
$U read https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHerc1667/representations/predictions/surfaces/20251217075048-surface-20260925170000-surface-m7-L2-prob.zarr 9.596 $Z $Y $X $NZ $NY $NX prob.raw --threads 16 --cache /mnt/scratch/vesuvius-offload/cache > /dev/null
AX=/mnt/scratch/pods/PHerc1667/unroll/axis.json
~/miniconda3/bin/python3 - <<PY
import numpy as np, json
from scipy import ndimage
z0,y0,x0,nz,ny,nx=$Z,$Y,$X,$NZ,$NY,$NX; sh=(nz,ny,nx)
cp=json.load(open("$AX"))["control_points"]
cz=np.array([c["z"] for c in cp])*.25; cy=np.array([c["y"] for c in cp])*.25; cx=np.array([c["x"] for c in cp])*.25; o=np.argsort(cz); cz,cy,cx=cz[o],cy[o],cx[o]
p=np.fromfile("prob.raw",np.uint8).reshape(sh).astype(np.float32); rg=np.zeros(sh,bool); Yg,Xg=np.mgrid[0:ny,0:nx].astype(np.float32)
for z in range(nz):
    ay=np.interp(z+z0,cz,cy)-y0; ax=np.interp(z+z0,cz,cx)-x0; dy=Yg-ay; dx=Xg-ax; r=np.sqrt(dy*dy+dx*dx)+1e-6; uy=dy/r; ux=dx/r; s=p[z]
    a=ndimage.map_coordinates(s,[Yg+uy,Xg+ux],order=1,mode="nearest"); b=ndimage.map_coordinates(s,[Yg-uy,Xg-ux],order=1,mode="nearest")
    rg[z]=(s>=64)&(s>=a)&(s>b)
rg.astype(np.uint8).tofile("ridge.raw")
gt=np.fromfile("gt.raw",np.uint8).reshape(sh); gs=(gt>=1)&(gt<=252); st=np.ones((3,3,3),bool)
g1=ndimage.binary_dilation(gs,structure=st); g8=ndimage.binary_dilation(gs,iterations=8)
for n,m in (("m7 >=128",p>=128),("m7 radial ridge",rg)):
    m1=ndimage.binary_dilation(m,structure=st); print("  %-18s fill %.3f recall %.4f precision %.4f"%(n,m.mean(),(gs&m1).sum()/max(1,gs.sum()),(m&g1).sum()/max(1,(m&g8).sum())),flush=True)
PY

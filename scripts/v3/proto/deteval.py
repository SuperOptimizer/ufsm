import numpy as np, json, sys
from scipy import ndimage
z0,y0,x0,nz,ny,nx=map(int,sys.argv[1:7]); sh=(nz,ny,nx)
cp=json.load(open("/vesuvius/ufsm/pods/PHercParis4_20261031/nocal/axis.json"))["control_points"]
cz=np.array([c["z"] for c in cp])*.25; cy=np.array([c["y"] for c in cp])*.25; cx=np.array([c["x"] for c in cp])*.25
o=np.argsort(cz); cz,cy,cx=cz[o],cy[o],cx[o]
p=np.fromfile("prob.raw",np.uint8).reshape(sh).astype(np.float32)
rg=np.zeros(sh,bool); Y,X=np.mgrid[0:ny,0:nx].astype(np.float32)
for z in range(nz):
    ay=np.interp(z+z0,cz,cy)-y0; ax=np.interp(z+z0,cz,cx)-x0; dy=Y-ay; dx=X-ax; r=np.sqrt(dy*dy+dx*dx)+1e-6; uy=dy/r; ux=dx/r; s=p[z]
    a=ndimage.map_coordinates(s,[Y+uy,X+ux],order=1,mode="nearest"); b=ndimage.map_coordinates(s,[Y-uy,X-ux],order=1,mode="nearest")
    rg[z]=(s>=64)&(s>=a)&(s>b)
rg.astype(np.uint8).tofile("ridge.raw")
gt=np.fromfile("gt.raw",np.uint8).reshape(sh); gs=(gt>=1)&(gt<=252); st=np.ones((3,3,3),bool)
g1=ndimage.binary_dilation(gs,structure=st); g8=ndimage.binary_dilation(gs,iterations=8)
def ev(name,m):
    m1=ndimage.binary_dilation(m,structure=st); rec=(gs&m1).sum()/max(1,gs.sum()); near=m&g8; prec=(m&g1).sum()/max(1,near.sum())
    print("  %-26s fill %.3f  recall %.4f  precision %.4f"%(name,m.mean(),rec,prec), flush=True)
ou=np.fromfile("ours.raw",np.uint8).reshape(sh); om=(ou>=1)&(ou<=252); m7=np.fromfile("m7.raw",np.uint8).reshape(sh)>=128
print("GT fill %.4f"%gs.mean())
ev("our spiral (wind96)",om); ev("m7 th0.2 >=128",m7); ev("m7 prob radial ridge",rg)
od=ndimage.binary_dilation(om,structure=st); rd=ndimage.binary_dilation(rg,structure=st)
ev("ours AND ridge (agree)",om&rd); ev("ours OR ridge",om|rg); ev("ridge NOT near ours",rg&~od)

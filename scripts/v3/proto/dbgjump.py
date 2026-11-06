import sys, os, json, glob, numpy as np, tifffile
from scipy import ndimage
from PIL import Image, ImageDraw
bd="C2"; z0,y0,x0,n=3328,1152,512,512; sgn=-1
gt=np.fromfile(bd+"/gt.raw",np.uint8).reshape(n,n,n); gs=(gt>=1)&(gt<=252); gtn=ndimage.maximum_filter(np.where(gs,gt,0),size=3)
ct=np.fromfile(bd+"/ctbox.raw",np.uint8).reshape(n,n,n); P=np.fromfile(bd+"/prob.raw",np.uint8).reshape(n,n,n)
cp=json.load(open("/mnt/scratch/pods/PHerc1667/unroll/axis.json"))["control_points"]; cz=np.array([c["z"] for c in cp])*.25; cy=np.array([c["y"] for c in cp])*.25; cx=np.array([c["x"] for c in cp])*.25; o=np.argsort(cz); cz,cy,cx=cz[o],cy[o],cx[o]
def th_(z,y,x): return np.arctan2(y+y0-np.interp(z+z0,cz,cy),x+x0-np.interp(z+z0,cz,cx))
best=None
for p in sorted(glob.glob(bd+"/paths/*")):
    if not os.path.exists(p+"/x.tif"): continue
    X,Y,Z=(tifffile.imread(p+"/"+c+".tif").astype(np.float32) for c in "xyz"); v=X>-1
    xi,yi,zi=(np.rint(A).astype(int) for A in (X,Y,Z)); v&=(xi>=0)&(yi>=0)&(zi>=0)&(xi<n)&(yi<n)&(zi<n)
    xi,yi,zi=xi[v],yi[v],zi[v]; c=gtn[zi,yi,xi]; m=c>0
    if m.sum()<50: continue
    th=th_(zi[m],yi[m],xi[m]); tr=np.arctan2(np.sin(th).mean(),np.cos(th).mean()); th=tr+np.angle(np.exp(1j*(th-tr)))
    s=np.mod(np.rint((c[m]-1)/18-sgn*th/(2*np.pi)),14).astype(int); d=np.bincount(s,minlength=14).argmax(); bad=(s!=d).sum()
    if best is None or bad>best[0]: best=(bad,p,xi[m],yi[m],zi[m],s,d)
bad,p,xi,yi,zi,s,d=best; zs=int(np.median(zi[s!=d])); print(p,bad,"z",zs)
def gray(a): return np.stack([a]*3,-1).astype(np.uint8)
A=gray(ct[zs]); B=gray(P[zs])
# GT sheets coloured by sheet id
yy,xx=np.nonzero(gs[zs]); th=th_(np.full(len(yy),zs,float),yy.astype(float),xx.astype(float))
q=np.mod(np.rint((gt[zs][yy,xx]-1)/18-sgn*th/(2*np.pi)),14).astype(int)
pal=np.array([[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],[145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,212],[0,128,128],[220,190,255],[170,110,40],[128,0,0]])
C=A.copy(); C[yy,xx]=pal[q]
k=np.abs(zi-zs)<=1; D=B.copy()
for X_,Y_,ok in zip(xi[k],yi[k],s[k]==d): D[max(0,Y_-1):Y_+2,max(0,X_-1):X_+2]=[0,255,0] if ok else [255,0,0]
E=A.copy()
for X_,Y_,ok in zip(xi[k],yi[k],s[k]==d): E[max(0,Y_-1):Y_+2,max(0,X_-1):X_+2]=[0,255,0] if ok else [255,0,0]
im=np.concatenate([np.concatenate([A,C],1),np.concatenate([E,D],1)],0)
Image.fromarray(im).resize((2048,2048),Image.NEAREST).save("dbg_jump_c2.png")

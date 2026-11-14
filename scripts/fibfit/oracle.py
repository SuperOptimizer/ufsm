# oracle same-sheet pairs from the published 1667 unwrap: points along its grid rows (H-like, along the sheet) and columns (V-like, z),
# spans 4 / 8 grid steps (~0.19 / 0.38 mm at 48 um), box-local level-2 native coords (box origin 0). keep: fraction of anchor vertices.
# zmod: only rows whose level-0 z // 1000 is even (a hold-out: score on the odd bands).
import sys, numpy as np, tifffile
pub='/vesuvius/ufsm/gt/pub1667-unwrap/20260612121456-on-20251217075048-2.399um.tifxyz'
SPANS=[int(v) for v in (sys.argv[4] if len(sys.argv)>4 else "4,8").split(",")]
out=sys.argv[1]; keep=float(sys.argv[2]); zmod=int(sys.argv[3]) if len(sys.argv)>3 else 0
X=tifffile.imread(pub+'/x.tif'); Y=tifffile.imread(pub+'/y.tif'); Z=tifffile.imread(pub+'/z.tif')
V=(Z>0)&(X>0)&(Y>0)&np.isfinite(Z)
P=np.stack([Z,Y,X],-1).astype(np.float32); P=(P+0.5)/4-0.5
rng=np.random.default_rng(1); recs=[]
for ax in (0,1):
    for s in SPANS:
        if ax==1: a=V[:,:-s]&V[:,s:]; A=P[:,:-s]; B=P[:,s:]
        else: a=V[:-s]&V[s:]; A=P[:-s]; B=P[s:]
        m=a&(rng.random(a.shape)<keep)
        if zmod: m&=((A[...,0]*4)//1000%2==0)
        d=np.linalg.norm(B-A,axis=-1); m&=d<s*20/4*1.3   # no tears
        r=np.concatenate([A[m],B[m],np.zeros((m.sum(),1),np.float32)],1); recs.append(r)
r=np.concatenate(recs).astype(np.float32); r.tofile(out); print(out,len(r),'pairs')

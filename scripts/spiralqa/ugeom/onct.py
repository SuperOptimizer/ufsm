"""audit ugeom #1: the fields' CT (level 4 raw) at every published vertex -> ctval.npy; coverage by CT value. Run in a pubcmp.sh OUT dir."""
import os, numpy as np, tifffile
P=os.environ.get('PUB','/vesuvius/ufsm/gt/pub1667-unwrap/20260612121456-on-20251217075048-2.399um.tifxyz')
Z=np.load('full/pub_dist.npz'); vi=Z['vi']; d=Z['d']; A=Z['area']
ct=np.memmap(os.environ.get('CT_RAW','/vesuvius/ufsm/gt/pub1667-cmp/ct_l4.raw'),dtype=np.uint8,mode='r',shape=(2318,952,952))   # 1667 level 4 (38.4 um) of the fields' CT
c=[]
for k in ('z','y','x'):
    a=tifffile.imread(f'{P}/{k}.tif').ravel()[vi]; c.append(np.clip((a/16).astype(np.int32),0,[2317,951,951][len(c)])); del a
v=np.asarray(ct[c[0],c[1],c[2]])
np.save('ctval.npy',v)
for t in (0,1,20,40,60,80,100,120):
    m=v>t; print(f'ct>{t}: area {A[m].sum()/A.sum():.3f} cov50 {A[m&(d<50)].sum()/A[m].sum():.3f} | below: cov50 {A[~m&(d<50)].sum()/max(A[~m].sum(),1):.3f}')
print(np.percentile(v,[1,5,10,25,50]))

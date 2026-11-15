"""audit ugeom #10: per-slab published identity (ramp removed) vs the slab coherence (slabcoh.txt) and the join's share; 1667 paths."""
import numpy as np
Z=np.load('/vesuvius/ufsm/gt/pub1667-cmp/full/pub_dist.npz'); X=np.load('b1667_aux.npz')
d=Z['d']; off=Z['off'].astype(np.int32); A=Z['area']; nc=Z['nearcut']; z=X['z']; u=X['u']
g=(d<100)&~nc
NB=64; ub=np.minimum((u*NB).astype(int),NB-1); ramp=np.zeros(NB,int)
for b in range(NB):
    v,c=np.unique(off[g&(ub==b)],return_counts=True); ramp[b]=v[np.argmax(c)]
e=off-ramp[ub]
coh={}
for l in open('/vesuvius/ufsm/gt/pub1667-cmp/slabcoh.txt'):
    f=l.split()
    if len(f)==3: coh[int(f[0][1:])]=(float(f[1]),float(f[2]))
js={}
import re
for l in open('/vesuvius/ufsm/unroll/PHerc1667-u1/logs/join.log'):
    m=re.match(r'\s+(\d+)\s+(\d+)\s+([\d.]+)\s+\+0\s+([\d.]+)\s+([\d.]+)$',l)
    if m: js[int(m[1])+2]=float(m[5])   # join slab index i = fit slab s(i+2)? (slab 0 = s002 per the command line)
sl=(z//512).astype(int)
rows=[]
print('slab zL0 area% cov50 same_ramp e+-1 |e|>=2 coh bias joinshare')
for k in range(sl.min(),sl.max()+1):
    m=sl==k
    if m.sum()<5000: continue
    mg=m&g
    c=coh.get(k,(np.nan,np.nan))
    r=(k,k*512,A[m].sum()/A.sum()*100,A[m&(d<50)].sum()/A[m].sum(),(e[mg]==0).mean(),(np.abs(e[mg])==1).mean(),(np.abs(e[mg])>=2).mean(),c[1],c[0],js.get(k,np.nan))
    rows.append(r); print('%3d %6d %4.1f %.3f %.3f %.3f %.3f %.3f %+.3f %.3f'%r)
R=np.array(rows); ok=np.isfinite(R[:,7])
from scipy.stats import spearmanr
print('spearman coherence vs same_ramp',spearmanr(R[ok,7],R[ok,4]))
print('spearman coherence vs cov50',spearmanr(R[ok,7],R[ok,3]))
hi=ok&(R[:,7]>=0.5); lo=ok&(R[:,7]<0.35)
w=R[:,2]
for nm,m in (('coh>=0.5',hi),('coh<0.35',lo)): print(nm,'slabs',m.sum(),'same_ramp',(R[m,4]*w[m]).sum()/w[m].sum(),'cov50',(R[m,3]*w[m]).sum()/w[m].sum(),'|e|>=2',(R[m,6]*w[m]).sum()/w[m].sum())
np.savetxt('slabz.tsv',R,fmt='%.4f',delimiter='\t',header='slab\tzL0\tarea_pct\tcov50\tsame_ramp\te1\te2\tcoh\tbias\tjoin_d025')

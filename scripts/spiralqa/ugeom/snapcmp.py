"""audit ugeom #7: unsnapped (d_raw.npz from pub_d_only.py) vs snapped wraps against the published surface; 1667 paths."""
import numpy as np
Z=np.load('/vesuvius/ufsm/gt/pub1667-cmp/full/pub_dist.npz'); R=np.load('d_raw.npz'); X=np.load('b1667_aux.npz')
ds=Z['d']; dr=R['d']; A=Z['area']; ids=Z['id']; idr=R['id']; pitch=X['pitch']; hole=X['hole']; u=X['u']
def cov(d,m,t): return A[m&(d<t)].sum()/A[m].sum()
allm=np.ones(len(ds),bool)
for t in (10,20,30,50,100): print(f'cov{t}: raw {cov(dr,allm,t):.4f} snap {cov(ds,allm,t):.4f}')
print('d_med raw %.1f snap %.1f'%(np.median(dr),np.median(ds)))
cl=(ds<100)&(dr<100)
print('on-sheet (both <100): d_med raw %.1f snap %.1f; p90 raw %.1f snap %.1f'%(np.median(dr[cl]),np.median(ds[cl]),np.percentile(dr[cl],90),np.percentile(ds[cl],90)))
print('nearest id same raw vs snap: %.4f'%(ids==idr).mean())
gain=(dr>=50)&(ds<50); loss=(dr<50)&(ds>=50)
print('snap: newly within 50 um area %.4f, lost %.4f'%(A[gain].sum()/A.sum(),A[loss].sum()/A.sum()))
for nm,key,ed in (('pitch',pitch,[-1e9,0,100,150,250,400,700,1e9]),('hole',hole,[0,16,64,1e9])):
    print('by',nm)
    for i in range(len(ed)-1):
        m=(key>=ed[i])&(key<ed[i+1])
        print(f'  {ed[i]:>8}-{ed[i+1]:<8} cov50 raw {cov(dr,m,50):.3f} snap {cov(ds,m,50):.3f}  cov20 raw {cov(dr,m,20):.3f} snap {cov(ds,m,20):.3f}  d_med on-sheet raw {np.median(dr[m&cl]):.1f} snap {np.median(ds[m&cl]):.1f}')

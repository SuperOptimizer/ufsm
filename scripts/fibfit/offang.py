# PHerc1667 vs the published unwrap (fibfit-20261114, DESIGN): run in order offang.py (pubv.npz: per published vertex theta, z, r, T,
# from pub1667-cmp/full/pub_dist.npz), then wpub.py MODEL_DIR [every] (sheet identity of a fit / count-field model's W). Work dir FIBFIT_WORK.
import json, os, sys, numpy as np, tifffile
pub='/vesuvius/ufsm/gt/pub1667-unwrap/20260612121456-on-20251217075048-2.399um.tifxyz'
d=np.load('/vesuvius/ufsm/gt/pub1667-cmp/full/pub_dist.npz')
vi=d['vi']; off=d['off']; dd=d['d']; T=d['T']; nc=d['nearcut']; oid=d['id']
X=tifffile.imread(pub+'/x.tif').ravel()[vi]; Y=tifffile.imread(pub+'/y.tif').ravel()[vi]; Z=tifffile.imread(pub+'/z.tif').ravel()[vi]
pts=json.load(open('/vesuvius/ufsm/unroll/PHerc1667-u1/axis.json'))['control_points']
a=np.array(sorted((p['z'],p['y'],p['x']) for p in pts))
cy=np.interp(Z,a[:,0],a[:,1]); cx=np.interp(Z,a[:,0],a[:,2])
th=np.degrees(np.arctan2(Y-cy,X-cx)); r=np.hypot(Y-cy,X-cx)*2.399/1000
m=(dd<100)&~nc
np.savez('' + os.environ.get('FIBFIT_WORK', '/vesuvius/ufsm/fibfit/1667/work/') + 'pubv.npz',th=th[m].astype(np.float32),z=Z[m].astype(np.float32),r=r[m].astype(np.float32),off=off[m],T=T[m],id=oid[m])
print('all same', (off[m]==0).mean())
bins=np.arange(-180,181,20)
for lo in bins[:-1]:
    s=m&(th>=lo)&(th<lo+20); o=off[s]
    print(f'{lo:5d} n {s.sum():9d} same {(o==0).mean():.3f} +1 {(o==1).mean():.3f} +2 {(o==2).mean():.3f} -1 {(o==-1).mean():.3f}')
for zl in range(0,37000,4000):
    s=m&(Z>=zl)&(Z<zl+4000)
    if s.sum()<1000: continue
    print('z',zl, s.sum(), ' '.join(f'{(off[s&(th>=lo)&(th<lo+40)]==0).mean():.2f}' for lo in range(-180,180,40)))
for rl in np.arange(3,12,1.0):
    s=m&(r>=rl)&(r<rl+1)
    if s.sum()<1000: continue
    print('r',rl, s.sum(), ' '.join(f'{(off[s&(th>=lo)&(th<lo+40)]==0).mean():.2f}' for lo in range(-180,180,40)))

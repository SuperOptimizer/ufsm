# W of a fit model (count-field ref or joined set slab) at the published surface's vertices; e = W - (T - u), u the angle fraction
# from theta = pi in the decreasing-angle direction (the published hand). A field of the right hand and sheet identity: e constant.
import json, os, sys, numpy as np
D=os.environ.get('FIBFIT_WORK', '/vesuvius/ufsm/fibfit/1667/work/')
mdir=sys.argv[1]; every=int(sys.argv[2]) if len(sys.argv)>2 else 20
m=json.load(open(mdir+'/model.json'))
L=m['L']; lz,ly,lx=m['lattice']; Dn=m['shape'][0]; h=m['h']; na=int(Dn*h)+1
delta=np.fromfile(mdir+'/delta.f32',dtype=np.float32).reshape(lz,ly,lx)
ax=np.fromfile(mdir+'/axis.f32',dtype=np.float32); cy,cx=ax[:na],ax[na:2*na]
pitch=np.array(m['pitch']); off=np.array(m['offset']); nk=len(pitch); kdz=m['knot_dz']; o=m['origin']
p=np.load(D+'pubv.npz'); sel=slice(None,None,every)
th=np.radians(p['th'][sel]).astype(np.float64); T=p['T'][sel].astype(np.float64); z0=p['z'][sel].astype(np.float64)
# need y,x: recompute from r, th and the axis.json (level 0)
pts=json.load(open('/vesuvius/ufsm/unroll/PHerc1667-u1/axis.json'))['control_points']; a=np.array(sorted((q['z'],q['y'],q['x']) for q in pts))
r0=p['r'][sel]*1000/2.399
Y0=np.interp(z0,a[:,0],a[:,1])+r0*np.sin(th); X0=np.interp(z0,a[:,0],a[:,2])+r0*np.cos(th)
z=(z0+0.5)/4-0.5-o[0]; y=(Y0+0.5)/4-0.5-o[1]; x=(X0+0.5)/4-0.5-o[2]
def lerp1(v,n,t): t=np.clip(t,0,n-1); i=np.minimum(t.astype(int),max(n-2,0)); f=t-i if n>1 else 0*t; return v[i]+f*(v[np.minimum(i+1,n-1)]-v[i])
def knot(v,t):
    k=np.clip(t/kdz,0,nk-1); k0=np.minimum(k.astype(int),max(nk-2,0)); f=k-k0; return v[k0]+f*(v[np.minimum(k0+1,nk-1)]-v[k0])
ok=(z>=0)&(z<=Dn*h-1)
def lat(z,y,x):
    q=[z/L,y/L,x/L]; dims=[lz,ly,lx]; c0=[];ct=[]
    for k in range(3):
        aa=np.clip(q[k],0,dims[k]-1); c=np.minimum(aa.astype(int),dims[k]-2); c0.append(c); ct.append(aa-c)
    s=0
    for c in range(8):
        bz,by,bx=c>>2&1,c>>1&1,c&1
        w=(ct[0] if bz else 1-ct[0])*(ct[1] if by else 1-ct[1])*(ct[2] if bx else 1-ct[2])
        s=s+w*delta[c0[0]+bz,c0[1]+by,c0[2]+bx]
    return s
acy=lerp1(cy,na,z); acx=lerp1(cx,na,z); tq=np.arctan2(y-acy,x-acx); rq=np.hypot(y-acy,x-acx)
W=rq/knot(pitch,z)+knot(off,z)+lat(z,y,x)-m['dir']*tq/(2*np.pi)
u=np.mod(-(tq-np.pi),2*np.pi)/(2*np.pi)
e=(W-(T-u))[ok]; thd=np.degrees(tq[ok])
c=np.angle(np.exp(2j*np.pi*e).mean())/(2*np.pi); k=np.round(e-c).astype(int); vals,cn=np.unique(k,return_counts=True); g=vals[np.argmax(cn)]
print(mdir,'dir',m['dir'],'n',ok.sum(),'sub-turn c',round(c,3),'share same',round((k==g).mean(),3),' +1',round((k==g+1).mean(),3),' -1',round((k==g-1).mean(),3),' |frac|<.25',round((np.abs(e-c-k)<0.25).mean(),3))
for lo in range(-180,180,30):
    s=(thd>=lo)&(thd<lo+30); print(f'  th {lo:5d} same {(k[s]==g).mean():.3f} +1 {(k[s]==g+1).mean():.3f} -1 {(k[s]==g-1).mean():.3f} +2 {(k[s]==g+2).mean():.3f} -2 {(k[s]==g-2).mean():.3f}')
rr=p['r'][sel][ok]; zz=z0[ok]
for lo in range(3,12):
    s=(rr>=lo)&(rr<lo+1)
    if s.sum()>1000: print(f'  r {lo:2d} mm n {s.sum():7d} same {(k[s]==g).mean():.3f} +1 {(k[s]==g+1).mean():.3f} -1 {(k[s]==g-1).mean():.3f}')
for lo in range(0,37000,4000):
    s=(zz>=lo)&(zz<lo+4000)
    if s.sum()>1000: print(f'  z {lo:5d} n {s.sum():7d} same {(k[s]==g).mean():.3f} +1 {(k[s]==g+1).mean():.3f} -1 {(k[s]==g-1).mean():.3f}')

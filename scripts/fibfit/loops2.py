# loops2.py FIELDS PREFIX H W Z,Z..: the phase-loop direction vote on field z slices (ufsm read pc / ps / w PREFIX{pc,ps,w}_Z.raw in FIBFIT_WORK/sl)
import json, os, sys, numpy as np
D=os.environ.get('FIBFIT_WORK', '/vesuvius/ufsm/fibfit/1667/work/') + 'sl/'
F,pre,H,W=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]); zs=[int(v) for v in sys.argv[5].split(',')]
ax=json.load(open(F+'/fields.json'))['axis']; h=2
for z in zs:
    pc=np.fromfile(D+f'{pre}pc_{z}.raw',np.uint8).reshape(H,W).astype(float); ps=np.fromfile(D+f'{pre}ps_{z}.raw',np.uint8).reshape(H,W).astype(float)
    w=np.fromfile(D+f'{pre}w_{z}.raw',np.uint8).reshape(H,W)
    m=((pc-128)/127)+1j*((ps-128)/127)
    cy,cx=ax[int((z+.5)*h)]; cy=cy/h-.5; cx=cx/h-.5
    pos=neg=n=0; byr=[]
    for r in np.arange(10,4000,3):
        ns=int(2*np.pi*r*1.5); t=np.linspace(-np.pi,np.pi,ns,endpoint=False)
        yy=np.round(cy+r*np.sin(t)).astype(int); xx=np.round(cx+r*np.cos(t)).astype(int)
        if yy.min()<0 or xx.min()<0 or yy.max()>=H or xx.max()>=W: break
        v=m[yy,xx]; ok=(abs(v)>0.3)&(w[yy,xx]>0); n+=1
        if ok.sum()<ns/2: continue
        ph=np.angle(v[ok])/(2*np.pi); d=np.diff(np.r_[ph,ph[:1]]); d-=np.round(d); s=round(d.sum()); pos+=s==1; neg+=s==-1; byr.append((r,s))
    print(z,'axis',round(cy),round(cx),'loops',n,'+',pos,'-',neg, 'by r:', [s for r,s in byr][::max(1,len(byr)//25)])

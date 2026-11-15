"""audit ugeom #2: refvsjoin.py DIR [boxes.tsv]: per-box GT scores of NAME.ref-cf.log vs NAME.joined.log (fit --model --codes --score-detail), pooled by radius."""
import re,glob,sys,os,numpy as np
d=sys.argv[1]
R=re.compile(r'fit score: (\d+) labelled voxels, gauge ([-\d.]+), winding ids right ([\d.]+), \|error\| < 0.08 turn ([\d.]+), \|error\| median ([\d.]+)')
P=re.compile(r'0.5 mm ([\d.]+) .*\| 4 mm ([\d.]+)')
RAD=re.compile(r'radius +([\d.]+)- *([\d.]+) mm +(\d+) voxels: ([\d. ]+)')
S={}
for f in glob.glob(f'{d}/*.log'):
    nm,m=os.path.basename(f)[:-4].split('.',1); t=open(f).read(); r=R.search(t)
    if not r: continue
    p=P.search(t)
    rad=[(float(a),int(n),[float(x) for x in v.split()]) for a,_,n,v in RAD.findall(t)]
    S.setdefault(nm,{})[m]=(int(r[1]),float(r[3]),float(r[4]),float(r[5]),float(p[1]) if p else np.nan,float(p[2]) if p else np.nan,rad)
noise=set()
bx=sys.argv[2] if len(sys.argv)>2 else None
if bx:
    for l in open(bx):
        if 'label-noise' in l: noise.add(l.split('\t')[0])
both=[k for k in S if 'ref-cf' in S[k] and 'joined' in S[k]]
print('boxes with both',len(both))
print('name  n(M)  ids ref->joined   err08 ref->joined   errmed ref->joined  pair05 ref->joined  pair4  noise')
for k in sorted(both,key=lambda k:S[k]['joined'][1]):
    a,b=S[k]['ref-cf'],S[k]['joined']
    print(f'{k:5} {a[0]/1e6:5.1f}  {a[1]:.3f}->{b[1]:.3f}  {a[2]:.3f}->{b[2]:.3f}  {a[3]:.3f}->{b[3]:.3f}  {a[4]:.3f}->{b[4]:.3f}  {a[5]:.3f}->{b[5]:.3f} {"N" if k in noise else ""}')
for nm,sel in (('all',both),('clean',[k for k in both if k not in noise])):
    if not sel: continue
    w=np.array([S[k]['joined'][0] for k in sel],float)
    for i,lab in ((1,'ids'),(2,'err08'),(3,'errmed'),(4,'pair05'),(5,'pair4')):
        ra=np.array([S[k]['ref-cf'][i] for k in sel]); jo=np.array([S[k]['joined'][i] for k in sel])
        print(f'{nm:5} {lab:6} ref {np.nansum(ra*w)/w.sum():.3f} joined {np.nansum(jo*w)/w.sum():.3f}  (box median ref {np.nanmedian(ra):.3f} joined {np.nanmedian(jo):.3f}; joined better in {(jo>ra).sum()}/{len(sel)})')
# pooled radius integer errors (joined, ref) clean boxes
for m in ('ref-cf','joined'):
    acc={}
    for k in both:
        if k in noise: continue
        for r0,n,v in S[k][m][6]:
            b=int(r0)  # 1 mm bins
            a=acc.setdefault(b,np.zeros(6)); a[0]+=n; a[1:]+=n*np.array(v)
    print('pooled by radius (clean boxes),',m,': r_mm  voxels(M)  -2 -1 0 +1 +2')
    for b in sorted(acc):
        a=acc[b]; print(f'   {b:2d}-{b+1:<2d} {a[0]/1e6:7.1f}  '+' '.join(f'{x/a[0]:.3f}' for x in a[1:]))

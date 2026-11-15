"""audit ugeom: the area budget of a pubcmp.sh OUT dir (full/pub_dist.npz, b1667_aux.npz, ctval.npy; d_raw.npz optional: the unsnapped wraps)."""
import numpy as np
import os
Z=np.load('full/pub_dist.npz'); X=np.load('b1667_aux.npz'); v=np.load('ctval.npy'); R=np.load('d_raw.npz') if os.path.exists('d_raw.npz') else None
A=Z['area']; d=Z['d']; jid=Z['id'].astype(int); off=Z['off'].astype(int); nc=Z['nearcut']; u=X['u']; hole=X['hole']; pitch=X['pitch']; z=X['z']; rad=X['rad']
tot=A.sum(); f=lambda m: A[m].sum()/tot*100
NB=64; ub=np.minimum((u*NB).astype(int),NB-1); g=(d<100)&~nc; ramp=np.zeros(NB,int)
for b in range(NB):
    vv,c=np.unique(off[g&(ub==b)],return_counts=True); ramp[b]=vv[np.argmax(c)]
e=off-ramp[ub]
cov=d<50; instrip=(jid>=0)&(jid<=39)
ct0=v==0
print('pub area %.1f cm2 = 100%%'%(tot*2.399**2*1e-8))
print('A. within 50 um of any of our 80 wraps: %.1f%%'%f(cov))
print('   of which on a strip wrap (w000-w039): %.1f%%  -> on a trimmed / non-strip wrap: %.1f%%'%(f(cov&instrip),f(cov&~instrip)))
print('   identity (not near the cut): right after the handedness ramp %.1f%%, off by 1 %.1f%%, off by >=2 %.1f%%, near the cut %.1f%%'%(f(cov&~nc&(e==0)),f(cov&~nc&(np.abs(e)==1)),f(cov&~nc&(np.abs(e)>=2)),f(cov&nc)))
miss=~cov
print('B. missed (d >= 50): %.1f%%'%f(miss))
print('   B1 on zero (masked) CT: %.1f%%'%f(miss&ct0))
m2=miss&~ct0
print('   B2 within 32 grid steps (~1.4 mm) of a hole / edge of the published surface: %.1f%%'%f(m2&(hole<32)))
m3=m2&(hole>=32)
print('   B3 negative local pitch (pub r not increasing with turn: crushed / axis off): %.1f%%'%f(m3&np.isfinite(pitch)&(pitch<0)))
m4=m3&~(np.isfinite(pitch)&(pitch<0))
print('   B4 in the handedness transition angles u 0.44-0.75: %.1f%%'%f(m4&(u>=0.44)&(u<0.75)))
m5=m4&~((u>=0.44)&(u<0.75))
print('   B5 scroll ends z<4000 or z>=34000: %.1f%%'%f(m5&((z<4000)|(z>=34000))))
m6=m5&~((z<4000)|(z>=34000))
print('   B6 rest: %.1f%%  (of which d 50-100: %.1f%%, >=100: %.1f%%)'%(f(m6),f(m6&(d<100)),f(m6&(d>=100))))
# reference rate for transition angles: cov50 in-angle vs out-angle on nonzero CT away from holes
for nm,m in (('u .44-.75',(u>=0.44)&(u<0.75)),('other u',~((u>=0.44)&(u<0.75)))):
    mm=m&~ct0&(hole>=32); print('  cov50 on CT, away from holes, %s: %.3f'%(nm,A[mm&cov].sum()/A[mm].sum()))
# geometric precision on-sheet
on=cov&(e==0)&~nc
w=A[on].astype(np.float64); dd=d[on]; o=np.argsort(dd); cw=np.cumsum(w[o])/w.sum(); cw[-1]=1
print('on-sheet, right identity: d area-median %.1f um, p90 %.1f'%(dd[o][np.searchsorted(cw,0.5)],dd[o][np.searchsorted(cw,0.9)]))
dr=(R['d'] if R is not None else d)[on]; o=np.argsort(dr); cw=np.cumsum(w[o])/w.sum(); cw[-1]=1; print('   unsnapped: median %.1f p90 %.1f'%(dr[o][np.searchsorted(cw,0.5)],dr[o][np.searchsorted(cw,0.9)]))
s=Z['s'][on]; print('   signed s median %.1f (ours outside pub > 0)'%np.median(s))

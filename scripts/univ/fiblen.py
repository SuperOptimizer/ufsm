import struct, sys, numpy as np
def load(p):
    b=open(p,'rb').read(); v,n,np_=struct.unpack('<3i',b[4:16]); o=16
    off=np.frombuffer(b,'<i4',n+1,o); o+=4*(n+1); o+=n
    return off,np.frombuffer(b,'<f4',3*np_,o).reshape(-1,3).astype(float)
off,pts=load(sys.argv[1]); seg=np.linalg.norm(np.diff(pts,axis=0),axis=1)*2.4/1000
cs=np.concatenate([[0],np.cumsum(seg)]); L=cs[off[1:]-1]-cs[off[:-1]]
print(f"n {len(L)} total {L.sum()/1000:.2f} m median {np.median(L):.2f} mean {L.mean():.2f} p90 {np.percentile(L,90):.2f} >=2mm {L[L>=2].sum()/L.sum():.2f}")

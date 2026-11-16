#!/usr/bin/env python3
# pair_ab.py CONTROL_LOG ARM_LOG: paired per-10-window differences (arm - control) of two --det 1 train logs over window blocks, 95 % block-bootstrap CIs, per head (DESIGN.md "Fully adaptive Muon" / "Muon update scaling").
import re,random,sys
K=[("loss",r" loss ([\d.]+)"),("phase",r"phase bce ([\d.]+)"),("rv",r"rv bce ([\d.]+)"),("normal",r"normal bce ([\d.]+)"),("wind4",r"wind4 bce ([\d.]+)"),("acc",r" acc ([\d.]+)"),("ink",r"ink bce ([\d.]+)"),("gn",r" gn ([\d.]+)")]
def parse(f):
    R={}
    for ln in open(f,errors='replace'):
        m=re.match(r"step +(\d+) lr",ln)
        if m: R[int(m.group(1))]={k:float(mm.group(1)) for k,rx in K if (mm:=re.search(rx,ln))}
    return R
C=parse(sys.argv[1]); A=parse(sys.argv[2]); random.seed(1)
def ci(d):
    bs=[]
    for _ in range(3000):
        x=[]
        while len(x)<len(d): i=random.randrange(0,max(1,len(d)-4)); x+=d[i:i+5]
        bs.append(sum(x)/len(x))
    bs.sort(); return bs[75],bs[2925]
for lo,hi in [(1,500),(501,1000),(1001,1600),(1601,2000)]:
    ss=[s for s in sorted(C) if lo<=s<=hi and s in A]
    if not ss: continue
    out=f"{lo:4d}-{hi:4d}:"
    for k,_ in K:
        d=[A[s][k]-C[s][k] for s in ss if k in A[s] and k in C[s]]
        if not d: continue
        c=sum(C[s][k] for s in ss)/len(ss)
        if k in ("loss","phase","wind4"):
            l,h=ci(d); out+=f" {k} {c:.4f} {sum(d)/len(d):+.4f}[{l:+.4f},{h:+.4f}]"
        else: out+=f" {k} {c:.4f} {sum(d)/len(d):+.4f}"
    print(out)
